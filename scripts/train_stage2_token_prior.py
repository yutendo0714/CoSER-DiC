from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

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


@torch.no_grad()
def evaluate(
    model: CausalTokenPrior,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int = 0,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    top1 = 0
    top5 = 0
    top64 = 0
    for batch_index, (targets_cpu,) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        targets = targets_cpu.to(device=device, dtype=torch.long, non_blocking=True)
        inputs = shifted_causal_inputs(targets, bos_token_id=model.bos_token_id)
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum")
        total_loss += float(loss.item())
        total_tokens += int(targets.numel())
        top = torch.topk(logits, k=min(64, logits.shape[-1]), dim=-1).indices
        target_expanded = targets.unsqueeze(-1)
        top1 += int((top[..., :1] == target_expanded).any(dim=-1).sum().item())
        top5 += int((top[..., : min(5, top.shape[-1])] == target_expanded).any(dim=-1).sum().item())
        top64 += int((top == target_expanded).any(dim=-1).sum().item())
    bits = total_loss / max(total_tokens, 1) / math.log(2.0)
    return {
        "loss_bits_per_token": bits,
        "perplexity": float(2.0**bits),
        "top1_hit": float(top1 / max(total_tokens, 1)),
        "top5_hit": float(top5 / max(total_tokens, 1)),
        "top64_hit": float(top64 / max(total_tokens, 1)),
        "tokens": float(total_tokens),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--output-dir", default="checkpoints/stage2_token_prior")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--wandb-mode", default="offline")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is not visible; stop and restart the container before training Stage 2 prior.")

    seed_everything(args.seed)
    if args.label_smoothing < 0.0 or args.label_smoothing >= 1.0:
        raise ValueError("--label-smoothing must be in [0, 1)")
    if args.grad_clip_norm <= 0:
        raise ValueError("--grad-clip-norm must be positive")
    if args.early_stop_patience < 0:
        raise ValueError("--early-stop-patience must be non-negative")
    if args.early_stop_min_delta < 0:
        raise ValueError("--early-stop-min-delta must be non-negative")
    token_payload = torch.load(args.tokens, map_location="cpu", weights_only=False)
    tokens = token_payload["tokens"].to(torch.long)
    if tokens.ndim != 3:
        raise ValueError("token export must contain tokens with shape [N, H, W]")
    n, h, w = (int(v) for v in tokens.shape)
    seq = tokens.reshape(n, h * w).contiguous()
    vocab_size = int(token_payload["codebook_size"])
    if int(seq.max().item()) >= vocab_size:
        raise ValueError("token IDs exceed exported codebook_size")

    order = torch.randperm(n)
    val_count = max(1, int(round(n * args.val_fraction)))
    val_idx = order[:val_count]
    train_idx = order[val_count:]
    train_tokens = seq[train_idx]
    val_tokens = seq[val_idx]
    train_loader = DataLoader(
        TensorDataset(train_tokens),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_tokens),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    cfg = CausalTokenPriorConfig(
        vocab_size=vocab_size,
        context_length=h * w,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    model = CausalTokenPrior(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp))

    run_name = args.run_name or f"{Path(args.tokens).parent.name}_causal_token_prior"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{run_name}.pt"
    summary_path = out_dir / f"{run_name}_summary.json"

    import wandb

    wandb_run = wandb.init(
        project="coserdic",
        name=run_name,
        mode=args.wandb_mode,
        config={**cfg.__dict__, **vars(args)},
    )

    best_val_bits = float("inf")
    best_step = 0
    stale_evals = 0
    stopped_early = False
    step = 0
    train_iter = iter(train_loader)
    progress = tqdm(total=args.max_steps, desc=run_name)
    while step < args.max_steps:
        try:
            (targets_cpu,) = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            (targets_cpu,) = next(train_iter)
        model.train()
        targets = targets_cpu.to(device=device, dtype=torch.long, non_blocking=True)
        inputs = shifted_causal_inputs(targets, bos_token_id=model.bos_token_id)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=bool(args.amp)):
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                label_smoothing=float(args.label_smoothing),
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip_norm))
        scaler.step(optimizer)
        scaler.update()

        step += 1
        train_bits = float(loss.item() / math.log(2.0))
        wandb_run.log(
            {
                "train/loss_bits_per_token": train_bits,
                "train/grad_norm": float(grad_norm),
                "train/lr": float(optimizer.param_groups[0]["lr"]),
                "step": step,
            },
            step=step,
        )
        progress.update(1)
        if step % args.eval_every == 0 or step == args.max_steps:
            metrics = evaluate(model, val_loader, device=device, max_batches=int(args.max_val_batches))
            wandb_run.log({f"val/{k}": v for k, v in metrics.items()} | {"step": step}, step=step)
            improved = metrics["loss_bits_per_token"] < best_val_bits - float(args.early_stop_min_delta)
            if improved:
                best_val_bits = metrics["loss_bits_per_token"]
                best_step = int(step)
                stale_evals = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": cfg.__dict__,
                        "token_export": args.tokens,
                        "step": step,
                        "val_metrics": metrics,
                        "train_args": vars(args),
                    },
                    ckpt_path,
                )
            else:
                stale_evals += 1
                if args.early_stop_patience and stale_evals >= int(args.early_stop_patience):
                    stopped_early = True
                    progress.write(
                        f"early stopping at step {step}: best={best_val_bits:.4f} bits/token at step {best_step}"
                    )
                    break
    progress.close()

    final_metrics = evaluate(model, val_loader, device=device, max_batches=int(args.max_val_batches))
    summary = {
        "tokens": args.tokens,
        "checkpoint": str(ckpt_path),
        "num_train_images": int(train_tokens.shape[0]),
        "num_val_images": int(val_tokens.shape[0]),
        "context_length": int(h * w),
        "vocab_size": vocab_size,
        "max_steps": args.max_steps,
        "completed_steps": int(step),
        "stopped_early": bool(stopped_early),
        "best_step": int(best_step),
        "label_smoothing": float(args.label_smoothing),
        "grad_clip_norm": float(args.grad_clip_norm),
        "early_stop_patience": int(args.early_stop_patience),
        "early_stop_min_delta": float(args.early_stop_min_delta),
        "max_val_batches": int(args.max_val_batches),
        "best_val_bits_per_token": best_val_bits,
        "final_val_bits_per_token": final_metrics["loss_bits_per_token"],
        "final_val_top1_hit": final_metrics["top1_hit"],
        "final_val_top5_hit": final_metrics["top5_hit"],
        "final_val_top64_hit": final_metrics["top64_hit"],
    }
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False))
    wandb_run.summary.update(summary)
    wandb_run.finish()

    payload = {
        "run_name": run_name,
        "date": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join([sys.executable, *sys.argv]),
        "summary": summary,
        "artifacts": {"checkpoint": str(ckpt_path), "summary": str(summary_path)},
    }
    write_run_doc(Path("docs/experiments") / f"{run_name}.md", payload)
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
