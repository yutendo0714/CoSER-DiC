from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class CausalTokenPriorConfig:
    vocab_size: int = 8192
    context_length: int = 64
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1


class CausalTokenPrior(nn.Module):
    """Small raster-causal Transformer prior for semantic VQ tokens."""

    def __init__(self, cfg: CausalTokenPriorConfig = CausalTokenPriorConfig()) -> None:
        super().__init__()
        if cfg.vocab_size <= 1:
            raise ValueError("vocab_size must be greater than 1")
        if cfg.context_length <= 0:
            raise ValueError("context_length must be positive")
        self.cfg = cfg
        self.bos_token_id = cfg.vocab_size
        self.token_embedding = nn.Embedding(cfg.vocab_size + 1, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.context_length, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.num_heads,
            dim_feedforward=int(round(cfg.d_model * cfg.mlp_ratio)),
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def config_dict(self) -> dict[str, object]:
        return asdict(self.cfg)

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        if input_tokens.ndim != 2:
            raise ValueError("input_tokens must have shape [B, L]")
        b, length = input_tokens.shape
        if length > self.cfg.context_length:
            raise ValueError("input length exceeds configured context_length")
        if torch.any((input_tokens < 0) | (input_tokens > self.bos_token_id)):
            raise ValueError("input token outside vocabulary/BOS range")
        positions = torch.arange(length, device=input_tokens.device).unsqueeze(0).expand(b, length)
        x = self.token_embedding(input_tokens) + self.position_embedding(positions)
        mask = torch.triu(torch.ones(length, length, device=input_tokens.device, dtype=torch.bool), diagonal=1)
        x = self.transformer(x, mask=mask)
        return self.head(self.norm(x))


def shifted_causal_inputs(target_tokens: torch.Tensor, *, bos_token_id: int) -> torch.Tensor:
    """Build teacher-forced inputs where position t sees tokens < t."""

    if target_tokens.ndim != 2:
        raise ValueError("target_tokens must have shape [B, L]")
    bos = torch.full((target_tokens.shape[0], 1), int(bos_token_id), device=target_tokens.device, dtype=torch.long)
    return torch.cat([bos, target_tokens[:, :-1].to(torch.long)], dim=1)


@torch.no_grad()
def topk_from_prefix(
    model: CausalTokenPrior,
    prefix: Sequence[int],
    *,
    topk: int,
    device: torch.device,
) -> list[int]:
    """Return decoder-rebuildable top-k candidates for the next token."""

    inputs = torch.tensor([[model.bos_token_id, *[int(v) for v in prefix]]], dtype=torch.long, device=device)
    logits = model(inputs)[:, -1, :]
    return torch.topk(logits, k=int(topk), dim=-1).indices.squeeze(0).detach().cpu().tolist()


@torch.no_grad()
def decoder_schedule_topk_indices(
    model: CausalTokenPrior,
    indices: torch.Tensor,
    *,
    topk: int,
    device: torch.device,
) -> torch.Tensor:
    """Build teacher-forced top-k rows in one causal forward pass.

    This is suitable for entropy-prior fitting and diagnostics. Actual
    bitstream encoding should use :func:`decoder_prefix_topk_indices` so the
    encoder uses exactly the same per-prefix computation as the decoder.
    """

    values = indices.detach().cpu().to(torch.long)
    if values.ndim == 2:
        batched = values.unsqueeze(0)
        squeeze_batch = True
    elif values.ndim == 3:
        batched = values
        squeeze_batch = False
    else:
        raise ValueError("indices must have shape [H, W] or [B, H, W]")
    seq = batched.reshape(batched.shape[0], -1).to(device=device, dtype=torch.long)
    inputs = shifted_causal_inputs(seq, bos_token_id=model.bos_token_id)
    logits = model(inputs)
    topk_indices = torch.topk(logits, k=int(topk), dim=-1).indices.detach().cpu()
    topk_grid = topk_indices.reshape(*batched.shape, int(topk))
    return topk_grid.squeeze(0) if squeeze_batch else topk_grid


@torch.no_grad()
def decoder_prefix_topk_indices(
    model: CausalTokenPrior,
    indices: torch.Tensor,
    *,
    topk: int,
    device: torch.device,
) -> torch.Tensor:
    """Build decoder-rebuildable top-k rows by replaying the prefix loop."""

    values = indices.detach().cpu().to(torch.long)
    if values.ndim == 2:
        batched = values.unsqueeze(0)
        squeeze_batch = True
    elif values.ndim == 3:
        batched = values
        squeeze_batch = False
    else:
        raise ValueError("indices must have shape [H, W] or [B, H, W]")

    rows_by_image: list[torch.Tensor] = []
    for image_tokens in batched.reshape(batched.shape[0], -1):
        prefix: list[int] = []
        rows: list[list[int]] = []
        for token in image_tokens.tolist():
            rows.append(topk_from_prefix(model, prefix, topk=topk, device=device))
            prefix.append(int(token))
        rows_by_image.append(torch.tensor(rows, dtype=torch.long).reshape(*batched.shape[1:], int(topk)))

    stacked = torch.stack(rows_by_image, dim=0)
    return stacked.squeeze(0) if squeeze_batch else stacked
