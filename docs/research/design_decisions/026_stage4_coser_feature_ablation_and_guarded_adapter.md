# Stage 4 CoSER Feature Ablation and Guarded Adapter Training

Date: 2026-06-28 JST  
Status: Active Stage 4 mainline diagnostic

## Decision

Stage 4 promotion must prove that decoded CoSER semantic/detail information is
used by the CoD-Lite conditioning path. Output-level RGB blending is not enough.

Add explicit decoder-side ablations to the Stage 4 evaluation and condition-stat
tools:

```text
semantic_latent_ablation:
  normal / zero / shuffle

detail_context_ablation:
  normal / zero / shuffle

condition_delta_ablation:
  normal / zero
```

These ablations do not change `actual_payload_bpp`; they only modify decoded
features inside the evaluation graph to audit whether the fixed adapter/gate is
using them.

## Implementation

Updated scripts:

```text
scripts/eval_stage4_cod_lite_adapter.py
scripts/analyze_stage4_cod_lite_condition_stats.py
scripts/train_stage4_cod_lite_adapter.py
```

Evaluation now supports:

```text
--semantic-latent-ablation zero|shuffle
--detail-context-ablation zero|shuffle
--condition-delta-ablation zero
```

The shuffle path uses a deterministic non-self permutation when the split has
more than one image.

Adapter training now supports guarded decoded-image losses:

```text
--ms-ssim-weight
--stage3-l1-guard-weight
--stage3-mse-guard-weight
--stage3-guard-margin
```

It also logs condition diagnostics during training:

```text
base_condition_l1
condition_l1_delta_vs_base
condition_cosine
pred_condition_std / target_condition_std
pred_condition_highfreq / target_condition_highfreq
```

## Smoke Checks

Validation:

```text
.venv/bin/python -m py_compile scripts/eval_stage4_cod_lite_adapter.py \
  scripts/analyze_stage4_cod_lite_condition_stats.py \
  scripts/train_stage4_cod_lite_adapter.py

.venv/bin/pytest tests/test_gencodec_backbone.py -q
```

Result:

```text
8 passed
```

Guarded adapter smoke:

```text
run:
  20260628_stage4_guarded_adapter_smoke_limit2

status:
  forward/backward succeeds with condition losses, MS-SSIM loss, and Stage3 guards
```

Base-condition ablation smoke:

```text
run:
  20260628_stage4_mainline_ablation_smoke_limit2

status:
  condition_delta_ablation=zero produces condition_delta_raw_l1=0
  condition_l1_delta_vs_base=0
```

## Full552 Condition-Space Ablation

Reference normal stats-match adapter:

```text
run:
  20260628_stage4_condition_stats_full552_statsmatch_ft2k

pred_to_target_l1:
  0.416470

pred_to_target_cosine:
  0.746259

pred_std:
  0.728248

pred_highfreq:
  0.234881

pred_condition_l1_win_rate:
  0.994565
```

Semantic latent zero:

```text
run:
  20260628_stage4_condition_stats_full552_ablate_semzero

pred_to_target_l1:
  0.429832

pred_to_target_cosine:
  0.724081

pred_std:
  0.697099

pred_highfreq:
  0.246789

pred_condition_l1_win_rate:
  0.983696
```

Detail context zero:

```text
run:
  20260628_stage4_condition_stats_full552_ablate_detailzero

pred_to_target_l1:
  0.417512

pred_to_target_cosine:
  0.744556

pred_std:
  0.727001

pred_highfreq:
  0.236025

pred_condition_l1_win_rate:
  0.998188
```

Interpretation:

```text
decoded 256-channel semantic latent is materially used by the adapter
current residual_grid/detail_codes context has only weak effect on condition recovery
detail context should not be removed, but it needs a Stage4-specific redesign
```

## Limit64 Image-Space Ablation

Same first 64 full552 samples, condition-space gate, unchanged payload:

```text
normal:
  PSNR / MS-SSIM: 21.6910 / 0.7395
  LPIPS / DISTS: 0.4969 / 0.3400
  condition_l1: 0.4587
  gate_mean: 0.4245

semantic zero:
  PSNR / MS-SSIM: 21.6291 / 0.7347
  LPIPS / DISTS: 0.5172 / 0.3521
  condition_l1: 0.4758
  gate_mean: 0.3210

detail zero:
  PSNR / MS-SSIM: 21.6961 / 0.7393
  LPIPS / DISTS: 0.4986 / 0.3410
  condition_l1: 0.4593
  gate_mean: 0.4256
```

Interpretation:

```text
image-space behavior agrees with condition-space diagnosis
semantic latent removal clearly hurts LPIPS/DISTS and condition_l1
detail-context removal is nearly neutral with the current adapter
```

## Research Consequence

Do not spend the next mainline effort on RGB output gates or fixed alpha.

Next priority:

```text
1. Train a stronger guarded adapter with condition imitation + image guards.
2. Add a diffusion-control detail projection head from the same transmitted detail payload.
3. Re-test full552 feature ablations after the detail control head is trained.
4. Only then revisit condition-space gate capacity or partial CoD-Lite unfreeze.
```

Update:

```text
The detail-aware condition-adapter implementation plus ft600/ft3000 full552
evaluation are now recorded in:

docs/research/design_decisions/027_stage4_detailaware_condition_adapter.md
```

This preserves the CoSER-DiC novelty axis:

```text
semantic VQ stream + detail residual stream
-> deterministic decoded CoSER feature state
-> CoSER-owned condition/control adapter
-> pretrained compression diffusion backbone
```
