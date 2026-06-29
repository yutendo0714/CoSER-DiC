# Stage 4 Detail-Control Branch Probe

Date: 2026-06-29

## Decision

The direct detail-control branch is useful infrastructure, but the first
ft800 probe is not a new promoted Stage 4 anchor.

It improves condition L1 slightly and preserves the no-extra-payload MVP
contract, but it does not materially improve full552 image metrics over the
previous fast8192 adapter. More importantly, detail zero/shuffle ablations
hurt by about the same amount as before. This means the branch has not yet made
the detail stream a substantially stronger diffusion-control signal.

Keep the implementation. Do not claim Stage 5 progress from this checkpoint.
Next work should strengthen how detail features control condition-space
residuals, not return to RGB postprocessing or fixed alpha tuning.

## Implementation

Added an optional condition-space detail branch:

```text
src/coserdic/models/gencodec_backbone.py
  CoSERToCoDLiteConditionPyramidAdapterConfig.detail_control_branch
  detail_control_out: hidden detail features -> CoD-Lite condition channels

scripts/train_stage4_cod_lite_adapter.py
  --detail-control-branch
```

The branch consumes the same decoded detail payload features:

```text
residual_grid_hat
normalized detail_codes
```

No image-specific side information is added. Therefore the evaluation
`actual_payload_bpp / paper_bpp` remains the Stage 3 semantic/detail payload
bpp.

The branch is zero-initialized, so non-strict initialization from older
checkpoints starts from the previous adapter behavior:

```text
missing keys:
  detail_control_out.weight
  detail_control_out.bias
unexpected keys:
  none
```

## Training Run

```text
run:
  20260629_stage4_fast8192_detailcontrol_ft800_b4

init:
  checkpoints/stage4_cod_lite_adapter/
  20260629_stage4_detailaware_ft600_fast8192_ft1500_b4.pt

checkpoint:
  checkpoints/stage4_cod_lite_adapter/
  20260629_stage4_fast8192_detailcontrol_ft800_b4.pt

train cache:
  results/bitstreams/stage3_training_cache_fast/
  20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4

batch:
  4

steps:
  800
```

Training summary:

```text
condition_l1_mean: 0.3957
base_condition_l1_mean: 0.5169
condition_l1_delta_vs_base_mean: -0.1212
condition_delta_raw_l1_mean: 0.5266
stage4_psnr_mean: 21.2397
lpips_mean: 0.4046
```

## Evaluation

Evaluation split:

```text
Kodak24 + CLIC2020 test428 + DIV2K val100
crop: 512x512
actual_payload_bpp / paper_bpp: 0.013999
```

Full552 result:

| checkpoint | PSNR | MS-SSIM | LPIPS | DISTS | condition_l1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fast8192 ft1500 | 21.3957 | 0.7186 | 0.4342 | 0.3022 | 0.4095 |
| semdrop ft800 | 21.2818 | 0.7186 | 0.4325 | 0.2982 | 0.4092 |
| detail-control ft800 | 21.3444 | 0.7160 | 0.4325 | 0.3013 | 0.4071 |

Interpretation:

```text
condition_l1 improves the most
LPIPS improves slightly over the previous fast8192 adapter
DISTS does not beat semantic dropout
PSNR/MS-SSIM remain below Stage 3 and below the previous raw fast8192 adapter
```

This is a condition-recovery improvement, not a full image-quality promotion.

## Detail-Use Ablation

All rows use the same checkpoint, Stage 3 bitstream, and frozen CoD-Lite
backbone. Only the decoded detail-context input to the adapter changes.

| detail context | PSNR | MS-SSIM | LPIPS | DISTS | condition_l1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| normal | 21.3444 | 0.7160 | 0.4325 | 0.3013 | 0.4071 |
| zero | 21.3575 | 0.7156 | 0.4377 | 0.3053 | 0.4081 |
| shuffle | 21.3339 | 0.7153 | 0.4360 | 0.3039 | 0.4083 |

Delta from normal:

```text
detail zero:
  LPIPS +0.0052
  DISTS +0.0041
  condition_l1 +0.0010

detail shuffle:
  LPIPS +0.0035
  DISTS +0.0026
  condition_l1 +0.0012
```

Compared with the previous fast8192 adapter, the detail ablation damage is
nearly unchanged. The direct branch did not yet increase detail-context
dependence in a meaningful way.

## Judgment

Promote this diagnosis:

```text
direct detail condition injection is architecturally valid
condition L1 alone can improve while final image metrics do not
detail stream remains under-used for diffusion control
the next improvement must alter conditioning quality, not final RGB output
```

Do not promote:

```text
fixed RGB blending
the ft800 detail-control checkpoint as Stage 5
external-baseline superiority
```

## Semantic-Dropout Follow-Up

After the first detail-control result, the branch was fine-tuned with
semantic-latent dropout and larger batch size:

```text
run:
  20260629_stage4_fast8192_detailcontrol_semdrop020_ft800_b8

init:
  checkpoints/stage4_cod_lite_adapter/
  20260629_stage4_fast8192_detailcontrol_ft800_b4.pt

semantic_latent_dropout_prob:
  0.20

batch:
  8
```

Batch 8 is feasible on the local RTX 4070 Ti SUPER and should be considered
for future short and medium Stage 4 probes when memory is available.

Full552 comparison:

| checkpoint | PSNR | MS-SSIM | LPIPS | DISTS | condition_l1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fast8192 ft1500 | 21.3957 | 0.7186 | 0.4342 | 0.3022 | 0.4095 |
| semdrop ft800 | 21.2818 | 0.7186 | 0.4325 | 0.2982 | 0.4092 |
| detail-control ft800 | 21.3444 | 0.7160 | 0.4325 | 0.3013 | 0.4071 |
| detail-control + semdrop b8 | 21.2325 | 0.7154 | 0.4315 | 0.2988 | 0.4069 |

Interpretation:

```text
LPIPS and condition_l1 are the best among these raw anchors
DISTS does not beat semantic dropout alone
PSNR/MS-SSIM regress further
this is not a promoted checkpoint
```

Detail ablation for the combined checkpoint:

| detail context | PSNR | MS-SSIM | LPIPS | DISTS | condition_l1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| normal | 21.2325 | 0.7154 | 0.4315 | 0.2988 | 0.4069 |
| zero | 21.2539 | 0.7152 | 0.4361 | 0.3023 | 0.4079 |
| shuffle | 21.2311 | 0.7149 | 0.4345 | 0.3009 | 0.4080 |

Delta from normal:

```text
detail zero:
  LPIPS +0.0046
  DISTS +0.0035
  condition_l1 +0.0010

detail shuffle:
  LPIPS +0.0029
  DISTS +0.0021
  condition_l1 +0.0010
```

This still does not materially increase detail-context dependence. The
combined run improves LPIPS by trading away fidelity, not by making the detail
stream a much stronger diffusion-control path.

## Next Mainline Work

Immediate probes:

```text
1. combine detail-control branch with semantic-latent dropout
   goal: reduce over-reliance on semantic latent and expose detail utility

2. add detail-specific condition supervision
   goal: make detail branch predict high-frequency / residual parts of the
   reference-native condition rather than letting the fusion trunk absorb it

3. test a stronger condition-space detail branch
   goal: residual blocks in condition space or FiLM-style modulation before
   condition residual output
```

Promotion requirement:

```text
normal full552 metrics must improve
detail zero/shuffle must hurt more than before
semantic zero/shuffle must still hurt clearly
actual_payload_bpp must remain unchanged unless a counted control stream is added
```
