# Stage 4 Condition Gate DISTS Probe

Date: 2026-06-29

## Decision

The DISTS-aware condition-space gate is a valid mainline probe, but the first
run is not promoted.

It stays inside the intended codec path:

```text
decoded CoSER semantic/detail payload
  -> Stage 4 adapter condition residual
  -> deterministic condition-space gate
  -> frozen CoD-Lite decoder
```

It does not use RGB output blending and it does not transmit image-specific
side information. Therefore `actual_payload_bpp / paper_bpp` remains the
Stage 3 semantic/detail payload.

The result is useful but not sufficient:

```text
the gate slightly recovers PSNR / MS-SSIM versus the raw adapter
but gives back LPIPS / DISTS
and its mean gate remains very close to the raw-adapter branch
```

So this is not a Stage 5 result and not an external-baseline win.

## Implementation

Added DISTS supervision to the condition gate trainer:

```text
scripts/train_stage4_cod_lite_condition_gate.py

--dists-weight
```

This is an image-domain training loss on the frozen CoD-Lite decoded output,
but the learned mechanism is still a condition-space gate:

```text
ungated_pred_cond = base_cond + adapter_condition_residual
gate = gate(decoded CoSER tensors, base_cond, adapter_condition_residual)
gated_pred_cond = base_cond + gate * adapter_condition_residual
stage4 = CoD-Lite(stage3, gated_pred_cond)
```

## Probe

Training:

```text
run:
  20260629_stage4_condition_gate_semdrop_detailcontrol_dists_probe600_b2ga2

adapter:
  checkpoints/stage4_cod_lite_adapter/
  20260629_stage4_fast8192_detailcontrol_semdrop020_ft800_b8.pt

train cache:
  results/bitstreams/stage3_training_cache_fast/
  20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4

steps:
  600

effective batch:
  4

gate range:
  0.55 to 1.0

loss additions:
  DISTS weight 0.4
```

Training summary:

| metric | value |
| --- | ---: |
| condition_l1_mean | 0.3949 |
| ungated_condition_l1_mean | 0.3941 |
| image_l1_mean | 0.0585 |
| LPIPS mean | 0.4070 |
| DISTS loss mean | 0.2752 |
| gate_mean_mean | 0.9482 |
| gate_min_mean | 0.8431 |
| gate_max_mean | 0.9889 |

Interpretation:

```text
the gate mean regularizer is too weak to prevent near-raw behavior
the trained gate mostly keeps the adapter residual active
```

## Limit64 Evaluation

Evaluation:

```text
run:
  20260629_stage4_condition_gate_semdrop_detailcontrol_dists_probe600_b2ga2_limit64_eval

manifest:
  strict 512 full552 feature cache

limit:
  first 64 samples
```

Comparison against the current semantic-dropout detail-control raw anchor:

| checkpoint | gate mean | PSNR | MS-SSIM | LPIPS | DISTS | condition_l1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw semdrop detail-control | 0.0000 | 21.6120 | 0.7393 | 0.4098 | 0.2965 | 0.4064 |
| DISTS gate probe600 | 0.9382 | 21.6376 | 0.7399 | 0.4139 | 0.3000 | 0.4075 |

The probe improves PSNR and MS-SSIM slightly, but worsens LPIPS, DISTS, and
condition L1 relative to the raw adapter. It is not promoted.

## Research Judgment

Promote:

```text
DISTS-capable condition-gate training infrastructure
evidence that condition-space gates can shift fidelity/perception tradeoff
```

Do not promote:

```text
this checkpoint as a Stage 4 anchor
gate mean regularization as the core solution
any fixed output-blend interpretation
```

## Next Mainline Direction

The next attempt should improve the condition adapter itself instead of only
placing a scalar gate on top of an imperfect condition residual.

Prioritized next moves:

```text
stronger multi-scale / condition-space adapter
channel-wise or group-wise condition residual control
condition statistics targets by useful channel groups
detail-control head that changes the adapter representation, not only its loss
joint adapter+gate training only after the base adapter becomes stronger
```

The key promotion rule remains:

```text
no RGB output blending as the method
same actual_payload_bpp unless extra controls are entropy-coded and counted
full552 evaluation before claiming a Stage 4/5 improvement
official CoD-Lite / CoD baselines remain the external target
```
