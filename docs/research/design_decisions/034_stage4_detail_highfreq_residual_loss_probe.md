# Stage 4 Detail High-Frequency Residual Loss Probe

Date: 2026-06-29

## Decision

The detail high-frequency residual loss is useful diagnostic infrastructure,
but the first probes are not promoted as Stage 4 anchors.

The loss successfully tests a mainline question:

```text
Can the decoded detail payload explain the high-frequency part of the
reference-vs-Stage3 CoD-Lite condition residual, without adding payload bits?
```

The answer from the first probes is:

```text
detail ablation sensitivity can be increased
but normal image quality regresses in LPIPS/DISTS
and the predicted detail high-frequency residual remains far below the target
```

Therefore, keep the implementation for future controlled ablations. Do not
claim this as Stage 5 progress, and do not run full552 promotion evaluation for
these checkpoints unless a later limit64 probe first beats the current anchors.

## Implementation

Added training-only detail residual supervision:

```text
scripts/train_stage4_cod_lite_adapter.py

--detail-highfreq-residual-weight
--detail-highfreq-kernel-size
```

Training computes:

```text
target_condition_residual = native_condition(reference) - native_condition(stage3)
pred_detail_residual = pred_condition_with_detail - pred_condition_with_zero_detail

loss = L1(
  local_highpass(pred_detail_residual),
  local_highpass(target_condition_residual)
)
```

This uses reference-derived condition only as a teacher during training.
Inference still uses only decoded CoSER semantic/detail features and fixed
model weights. No image-specific side information is transmitted, so
`actual_payload_bpp / paper_bpp` is unchanged.

## Probes

All probes use:

```text
train cache:
  results/bitstreams/stage3_training_cache_fast/
  20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4

initial detail-control checkpoint:
  checkpoints/stage4_cod_lite_adapter/
  20260629_stage4_fast8192_detailcontrol_ft800_b4.pt

batch:
  8

steps:
  800

backbone:
  official CoD-Lite bpp0.0156 wrapper

payload:
  unchanged Stage 3 semantic/detail actual bitstream
```

### w0.5 From Detail-Control

```text
run:
  20260629_stage4_fast8192_detailhfres_w05_ft800_b8
```

Training summary:

| metric | value |
| --- | ---: |
| condition_l1_mean | 0.3966 |
| detail_highfreq_residual_loss_mean | 0.2895 |
| detail_highfreq_residual_pred_l1_mean | 0.0136 |
| detail_highfreq_residual_target_l1_mean | 0.2908 |
| stage4_psnr_mean | 21.1978 |
| lpips_mean | 0.4050 |

Interpretation:

```text
the target residual high-frequency energy is large
the learned detail contribution remains much smaller
the loss does not yet make the detail branch explain the target residual
```

### w0.1 From Semantic-Dropout Detail-Control

```text
run:
  20260629_stage4_fast8192_detailcontrol_semdrop020_detailhfres_w01_ft800_b8

semantic_latent_dropout_prob:
  0.20
```

Training summary:

| metric | value |
| --- | ---: |
| condition_l1_mean | 0.3972 |
| detail_highfreq_residual_loss_mean | 0.2908 |
| detail_highfreq_residual_pred_l1_mean | 0.0088 |
| detail_highfreq_residual_target_l1_mean | 0.2908 |
| stage4_psnr_mean | 21.0948 |
| lpips_mean | 0.4058 |
| semantic_latent_drop_fraction_mean | 0.2014 |

Interpretation:

```text
weaker weight plus semantic dropout does not solve the residual gap
the model still mostly avoids using detail for condition high-frequency energy
```

## Limit64 Evaluation

Evaluation:

```text
manifest:
  results/bitstreams/stage3_uniform_residual/
  20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl

limit:
  first 64 strict 512 samples

actual_payload_bpp:
  unchanged Stage 3 semantic/detail payload
```

| checkpoint | PSNR | MS-SSIM | LPIPS | DISTS | condition_l1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| detail-control + semdrop b8 | 21.6120 | 0.7393 | 0.4098 | 0.2965 | 0.4064 |
| detail-HF w0.5 | 21.7426 | 0.7392 | 0.4157 | 0.3014 | 0.4075 |
| semdrop + detail-HF w0.1 | 21.6807 | 0.7384 | 0.4170 | 0.3021 | 0.4074 |

The detail-HF loss improves PSNR on limit64, but it hurts the perceptual
metrics that Stage 4 is supposed to improve.

## Detail Ablation

For the w0.5 checkpoint:

| detail context | PSNR | MS-SSIM | LPIPS | DISTS | condition_l1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| normal | 21.7426 | 0.7392 | 0.4157 | 0.3014 | 0.4075 |
| zero | 21.6684 | 0.7386 | 0.4203 | 0.3012 | 0.4113 |
| shuffle | 21.6757 | 0.7361 | 0.4205 | 0.3046 | 0.4122 |

This is the one useful signal:

```text
detail zero/shuffle now hurts condition_l1 and LPIPS more clearly
```

But the normal checkpoint itself is worse than the semdrop anchor, so this is
not a promotion.

## Research Judgment

Promote the diagnosis:

```text
detail stream can be made more consequential for condition prediction
global/high-frequency condition residual supervision is too blunt
detail control needs a better target and/or architecture
```

Do not promote:

```text
detail-HF residual loss as the current main Stage 4 method
the w0.5 or w0.1 checkpoint as a Stage 5 candidate
any external-baseline win
```

## Next Mainline Direction

The next attempt should avoid forcing detail to match the whole condition
high-frequency residual. Better targets:

```text
condition-space channel groups that actually correlate with texture/detail
perceptual feature residuals after frozen CoD-Lite decode
learned deterministic condition gate over adapter residual channels
diffusion-control detail head trained with image/perceptual guard first,
then condition residual diagnostics second
```

The key requirement remains:

```text
same transmitted semantic/detail payload
no decoder-time reference leakage
actual_payload_bpp unchanged unless a future control stream is explicitly
entropy-coded and counted
```
