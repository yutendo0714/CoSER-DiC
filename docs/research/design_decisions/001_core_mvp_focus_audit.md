# Core MVP Focus Audit

Updated: 2026-06-27 JST

This audit translates the MVP-v0 policy into repository constraints.

## Fixed Core

- RD-aware semantic VQ token stream
- detail residual latent stream
- semantic-conditioned residual entropy model
- auxiliary reconstruction
- compression-oriented one-step diffusion reconstruction
- actual entropy-coded bitstream and actual bpp reporting

## Default Initialization

Use only one pretrained component by default:

```text
CoD_Lite_pretrain.pt -> diffusion decoder backbone
```

Everything else is scratch-trained CoSER-DiC:

- semantic encoder/tokenizer
- VQ codebook
- semantic categorical entropy model
- semantic auxiliary decoder
- detail residual encoder
- detail Gaussian entropy model
- semantic-conditioned residual prior
- joint auxiliary decoder
- CoSER conditioning adapter / FiLM / control branch

## Repository Constraints

- `src/coserdic` is the only CoSER-DiC implementation root.
- `external/repos` is for baselines and references only.
- `external/pretrained` is for downloaded third-party weights only.
- Main results must use `compress()` and `decompress()`.
- Estimated bpp is diagnostic only.
- No original-derived masks, captions, OCR maps, segmentation maps, or
  importance maps may reach the decoder unless transmitted in the bitstream.

## Deferred Until MVP Completes

- VLM-DPO
- OCR/face identity losses
- adaptive quantization
- advanced variable-rate control
- full autoregressive token prior
- Mamba/SSM entropy models
- large human preference study
- complex semantic-critical benchmark construction

