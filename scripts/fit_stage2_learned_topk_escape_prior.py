from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from coserdic.entropy import TopKEscapeHuffmanCode, count_topk_escape_events
from coserdic.models import CausalTokenPrior, CausalTokenPriorConfig, shifted_causal_inputs
from coserdic.utils import seed_everything


def write_run_doc(path: Path, payload: dict) -> None:
    lines = [
        f"# {payload['run_name']}",
        "",
        f"Date: {payload['date']}",
        "",
        "## Command",
        "",
        "```bash",
        payload["command"],
        "```",
        "",
        "## Summary",
        "",
    ]
    for key, value in payload["summary"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Artifacts", ""])
    for key, value in payload["artifacts"].items():
        lines.append(f"- {key}: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def load_token_prior(path: Path, device: torch.device) -> CausalTokenPrior:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = CausalTokenPriorConfig(**checkpoint["config"])
    model = CausalTokenPrior(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.no_grad()
def topk_for_targets(model: CausalTokenPrior, targets: torch.Tensor, *, topk: int) -> torch.Tensor:
    inputs = shifted_causal_inputs(targets, bos_token_id=model.bos_token_id)
    logits = model(inputs)
    return torch.topk(logits, k=int(topk), dim=-1).indices.detach().cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--token-prior-checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/stage2_learned_entropy")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--event-smoothing-count", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--wandb-mode", default="offline")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is not visible; stop and restart the container before fitting learned Stage 2 prior.")

    seed_everything(args.seed)
    token_payload = torch.load(args.tokens, map_location="cpu", weights_only=False)
    tokens = token_payload["tokens"].to(torch.long)
    if tokens.ndim != 3:
        raise ValueError("token export must contain tokens with shape [N, H, W]")
    if args.max_images:
        tokens = tokens[: int(args.max_images)]
    n, h, w = (int(v) for v in tokens.shape)
    seq = tokens.reshape(n, h * w).contiguous()
    vocab_size = int(token_payload["codebook_size"])
    if int(args.topk) >= vocab_size:
        raise ValueError("--topk must be smaller than codebook_size")

    model = load_token_prior(Path(args.token_prior_checkpoint), device)
    if model.cfg.vocab_size != vocab_size:
        raise ValueError("token prior vocab_size does not match token export")
    if model.cfg.context_length != h * w:
        raise ValueError("token prior context_length does not match token export")

    loader = DataLoader(
        TensorDataset(seq),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    event_counts = torch.zeros(int(args.topk) + 1, dtype=torch.long)
    total_tokens = 0
    for (targets_cpu,) in tqdm(loader, desc="fit top-k events"):
        targets = targets_cpu.to(device=device, dtype=torch.long, non_blocking=True)
        topk_indices = topk_for_targets(model, targets, topk=args.topk)
        event_counts += count_topk_escape_events(
            targets_cpu,
            topk_indices,
            codebook_size=vocab_size,
            topk=int(args.topk),
        )
        total_tokens += int(targets_cpu.numel())

    code = TopKEscapeHuffmanCode.from_event_counts(
        event_counts,
        codebook_size=vocab_size,
        topk=int(args.topk),
        smoothing_count=int(args.event_smoothing_count),
    )

    payload_bytes = 0
    unpadded_bits = 0
    for (targets_cpu,) in tqdm(loader, desc="measure top-k payload"):
        targets = targets_cpu.to(device=device, dtype=torch.long, non_blocking=True)
        topk_indices = topk_for_targets(model, targets, topk=args.topk)
        for image_index in range(targets_cpu.shape[0]):
            target_image = targets_cpu[image_index].reshape(h, w)
            topk_image = topk_indices[image_index].reshape(h, w, int(args.topk))
            bits = code.encoded_bits(target_image, topk_image)
            payload_bytes += (bits + 7) // 8
            unpadded_bits += bits

    run_name = args.run_name or f"{Path(args.token_prior_checkpoint).stem}_topk{args.topk}_escape_huffman"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    prior_path = out_dir / "learned_topk_escape_huffman_prior.json"
    summary_path = out_dir / "summary.json"

    prior_payload = code.to_dict()
    prior_payload.update(
        {
            "token_prior_checkpoint": str(args.token_prior_checkpoint),
            "token_export": str(args.tokens),
            "event_counts": [int(v) for v in event_counts.tolist()],
            "event_smoothing_count": int(args.event_smoothing_count),
        }
    )
    prior_path.write_text(json.dumps(prior_payload, indent=2, allow_nan=False))

    escape_count = int(event_counts[-1].item())
    event_probs = event_counts.to(torch.float64) / max(int(event_counts.sum().item()), 1)
    nonzero = event_probs[event_probs > 0]
    event_entropy = float((-(nonzero * torch.log2(nonzero))).sum().item())
    image_pixels = int(args.image_size) * int(args.image_size)
    summary = {
        "tokens": str(args.tokens),
        "token_prior_checkpoint": str(args.token_prior_checkpoint),
        "prior": str(prior_path),
        "num_images": n,
        "token_shape": [h, w],
        "codebook_size": vocab_size,
        "topk": int(args.topk),
        "event_smoothing_count": int(args.event_smoothing_count),
        "total_tokens": int(total_tokens),
        "top1_hit_rate": float(event_counts[0].item() / max(total_tokens, 1)),
        "top5_hit_rate": float(event_counts[: min(5, int(args.topk))].sum().item() / max(total_tokens, 1)),
        "topk_hit_rate": float(1.0 - escape_count / max(total_tokens, 1)),
        "escape_rate": float(escape_count / max(total_tokens, 1)),
        "event_entropy_bits": event_entropy,
        "mean_unpadded_bits_per_token": float(unpadded_bits / max(total_tokens, 1)),
        "mean_payload_bytes_per_image": float(payload_bytes / max(n, 1)),
        "mean_payload_bpp_image_size": float(payload_bytes * 8 / max(n * image_pixels, 1)),
        "fixed_bits_payload_bpp_image_size": float(math.ceil(math.log2(vocab_size)) * h * w / image_pixels),
    }
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False))

    import wandb

    wandb_run = wandb.init(
        project="coserdic",
        name=run_name,
        mode=args.wandb_mode,
        config={**summary, "event_counts": [int(v) for v in event_counts.tolist()]},
    )
    wandb_run.summary.update(summary)
    wandb_run.finish()

    payload = {
        "run_name": run_name,
        "date": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join([sys.executable, *sys.argv]),
        "summary": summary,
        "artifacts": {"prior": str(prior_path), "summary": str(summary_path)},
    }
    write_run_doc(Path("docs/experiments") / f"{run_name}.md", payload)
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
