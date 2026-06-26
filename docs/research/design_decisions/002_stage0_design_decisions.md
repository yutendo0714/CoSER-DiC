# Stage 0 Design Decisions

Updated: 2026-06-27 JST

## Keep CoSER-DiC Simple at MVP

The user explicitly noted that simpler model designs often improve performance
more reliably. The MVP should therefore avoid adding all possible modern
components at once.

Locked MVP:

- semantic encoder + differentiable VQ
- semantic entropy model
- semantic auxiliary decoder
- detail residual encoder
- semantic-conditioned residual hyperprior
- joint auxiliary decoder
- one-step/few-step diffusion adapter
- actual bitstream round-trip

Deferred:

- VLM-DPO
- OCR/face losses as training losses
- original-derived masks/captions/control maps
- full autoregressive token prior
- Mamba prior
- large multi-step diffusion sampling

## Actual bpp Comes Before Paper Metrics

All evaluation scripts should support forward-only diagnostics, but main results
must call `compress()` and `decompress()`. The bpp denominator is the original
unpadded image size:

```text
actual_bpp = 8 * len(bitstream) / (H * W)
```

Header overhead is included. Model weights and metric logs are excluded.

## Recommended First Implementation Variant

Stage 1 should prioritize a stable semantic tokenizer:

- downsample factor 32
- codebook size 8192
- embedding dim 256
- EMA VQ plus straight-through estimator
- auxiliary decoder with L1 + MS-SSIM + LPIPS
- codebook usage logging

Stage 2 can import RDVQ-like soft assignment rate feedback, but the first
actual stream can use a simple learned categorical prior before introducing
heavier context models.

Stage 3 residual:

- 64 latent channels initially
- Gaussian conditional with hyperprior
- semantic FiLM/context for mean and scale
- stronger residual rate penalty than semantic rate penalty

Diffusion:

- prefer CoD-Lite initialization if checkpoint terms are compatible;
- keep StableCodec as a baseline/fallback reference only;
- freeze diffusion backbone first;
- train adapters/FiLM/conditioning;
- keep auxiliary L1 anchor to prevent semantic drift.
