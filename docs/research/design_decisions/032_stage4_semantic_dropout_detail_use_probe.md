# Stage 4 Semantic-Dropout Detail-Use Probe

Date: 2026-06-29

## Decision

Semantic-latent dropout is useful as a regularizer, but it does not by itself
solve the current Stage 4 bottleneck.

It slightly improves full552 LPIPS/DISTS at unchanged actual payload bpp, while
reducing PSNR. It also does not materially increase detail-context ablation
sensitivity. Therefore, do not promote this checkpoint as the new main raw
anchor. Keep the infrastructure, and move the mainline toward a stronger
diffusion-control detail head / condition-space detail injection.

## Implementation

The Stage 4 adapter trainer now supports decoder-feature dropout:

```text
scripts/train_stage4_cod_lite_adapter.py

--semantic-latent-dropout-prob
--detail-context-dropout-prob
```

This is training-only corruption. Inference still uses decoded CoSER features
from the transmitted semantic/detail payload and does not transmit any extra
side information.

## Run

```text
train:
  20260629_stage4_fast8192_ft1500_semdrop020_ft800_b4

init:
  checkpoints/stage4_cod_lite_adapter/
  20260629_stage4_detailaware_ft600_fast8192_ft1500_b4.pt

checkpoint:
  checkpoints/stage4_cod_lite_adapter/
  20260629_stage4_fast8192_ft1500_semdrop020_ft800_b4.pt

train cache:
  results/bitstreams/stage3_training_cache_fast/
  20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4

dropout:
  semantic_latent_dropout_prob = 0.20
  detail_context_dropout_prob = 0.00
```

Evaluation split:

```text
Kodak24 + CLIC2020 test428 + DIV2K val100
crop: 512x512
actual_payload_bpp / paper_bpp: 0.013999
```

## Full552 Result

Baseline is the previous fast8192 ft1500 checkpoint.

| checkpoint | PSNR | MS-SSIM | LPIPS | DISTS | condition_l1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fast8192 ft1500 | 21.3957 | 0.7186 | 0.4342 | 0.3022 | 0.4095 |
| + semantic dropout ft800 | 21.2818 | 0.7186 | 0.4325 | 0.2982 | 0.4092 |

Interpretation:

```text
LPIPS improves by about 0.0017
DISTS improves by about 0.0040
condition_l1 is essentially unchanged
PSNR drops by about 0.114 dB
```

This is a useful perceptual regularization signal, not a clean promotion.

## By Dataset

Semantic-dropout full552 result:

| dataset | count | PSNR | MS-SSIM | LPIPS | DISTS | condition_l1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Kodak | 24 | 21.0252 | 0.7032 | 0.4708 | 0.3173 | 0.4243 |
| CLIC2020 test | 428 | 21.6445 | 0.7308 | 0.4204 | 0.2946 | 0.4049 |
| DIV2K val | 100 | 19.7912 | 0.6699 | 0.4751 | 0.3088 | 0.4239 |

Artifact:

```text
results/analysis/stage4_semantic_dropout/
20260629_stage4_fast8192_semdrop020_ft800_full552_by_dataset.json
```

## Feature-Use Ablation

All rows use the same Stage 3 payload and the same frozen CoD-Lite backbone.
Only decoder-side adapter inputs are ablated.

| ablation | PSNR | MS-SSIM | LPIPS | DISTS | condition_l1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| normal | 21.2818 | 0.7186 | 0.4325 | 0.2982 | 0.4092 |
| semantic latent zero | 21.1284 | 0.7102 | 0.4359 | 0.3084 | 0.4118 |
| semantic latent shuffle | 20.8946 | 0.7091 | 0.4418 | 0.3036 | 0.4192 |
| detail context zero | 21.3055 | 0.7184 | 0.4372 | 0.3017 | 0.4100 |
| detail context shuffle | 21.2830 | 0.7181 | 0.4356 | 0.3004 | 0.4101 |

Comparison against the previous fast8192 ablation:

```text
semantic ablation still hurts clearly
detail ablation still hurts only mildly
detail sensitivity did not materially increase
```

## Judgment

Promote the following diagnosis:

```text
semantic latent remains the dominant CoSER-to-CoD-Lite condition signal
semantic dropout can improve perceptual metrics slightly
detail context is still under-used as a diffusion-control signal
dropout alone is not enough to make the detail stream do the Stage 4 work
```

Do not promote this as Stage 5 or as an external-baseline win.

## Next Mainline Step

The next implementation should target the information flow directly:

```text
same transmitted detail payload
-> stronger detail-control projection / condition-space detail branch
-> CoSER condition adapter
-> frozen CoD-Lite / CoD backbone
```

Recommended first version:

```text
add a learned detail-control head from residual_grid_hat + detail_codes
inject it into condition-space residual blocks
keep actual_payload_bpp unchanged
require full552 improvement and stronger detail ablation sensitivity
```

After that, test whether semantic dropout remains useful as a regularizer in
the stronger detail-control architecture.
