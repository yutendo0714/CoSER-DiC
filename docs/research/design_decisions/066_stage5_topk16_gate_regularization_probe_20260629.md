# 066 Stage 5 Top-k16 Gate Regularization Probe

Date: 2026-06-29 JST

## Decision

Do not promote the new top-k16 smooth/mid gate regularization runs as Stage 5
anchors.

Keep the existing top-k16 curve point and the top-k16-specific perceptual gate
as the useful low-rate references:

```text
balanced k32 gate reused at top-k16:
  safer low-rate curve point

top-k16-specific gate:
  slightly better LPIPS/DISTS at equal actual bpp
  slightly worse PSNR/MS-SSIM
```

The gate regularization implementation should remain, because it is a useful
mainline training control for deterministic condition-space gates. The result
shows that further gains are unlikely to come from only smoothing or mean-
regularizing the current gate. The next mainline work should target adapter
condition recovery, control representation, and diffusion-friendly detail
features.

## Implementation

Added optional regularizers to:

```text
scripts/train_stage4_cod_lite_condition_gate.py
```

New arguments:

```text
--gate-deviation-weight
--gate-tv-weight
```

These losses are applied to the deterministic CoD-Lite condition gate, not to
the final RGB image:

```text
gate_deviation_loss = mean((gate - gate_mean_target)^2)
gate_tv_loss        = mean(|spatial gate gradients|)
```

When the weights are zero, training behavior is unchanged.

## Results

Strict full552 protocol:

```text
Kodak24 + CLIC2020 test428 + DIV2K val100
512x512 center crop
actual_payload_bpp from entropy-coded CoSER semantic/detail plus counted
control payload
```

| point | bpp | control bpp | PSNR | MS-SSIM | LPIPS | DISTS | condition L1 | post-gate mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage4 no-extra | 0.013999 | 0.000000 | 21.3043 | 0.7166 | 0.4080 | 0.2780 | 0.4104 | 0.0000 |
| top-k16 reuse k32 gate | 0.014977 | 0.000978 | 21.2341 | 0.7144 | 0.3833 | 0.2662 | 0.4020 | 1.0043 |
| top-k16-specific gate | 0.014977 | 0.000978 | 21.2214 | 0.7139 | 0.3818 | 0.2650 | 0.4023 | 1.0166 |
| top-k16 smoothguard | 0.014977 | 0.000978 | 21.2424 | 0.7150 | 0.3866 | 0.2674 | 0.4021 | 0.9925 |
| top-k16 midguard | 0.014977 | 0.000978 | 21.2357 | 0.7147 | 0.3846 | 0.2666 | 0.4021 | 1.0034 |
| top-k32 balanced | 0.015484 | 0.001485 | 21.2378 | 0.7146 | 0.3825 | 0.2660 | 0.4013 | 1.0046 |

Delta versus top-k16 reuse k32 gate:

| point | PSNR | MS-SSIM | LPIPS | DISTS | condition L1 |
|---|---:|---:|---:|---:|---:|
| top-k16-specific | -0.0127 | -0.0005 | -0.0014 | -0.0012 | +0.0003 |
| top-k16 smoothguard | +0.0083 | +0.0006 | +0.0033 | +0.0013 | +0.0001 |
| top-k16 midguard | +0.0016 | +0.0002 | +0.0013 | +0.0004 | +0.0001 |

## Interpretation

The smoothguard run confirms that gate regularization does what it should:
it narrows the gate range and improves PSNR/MS-SSIM versus the reused-gate
top-k16 point. However, this also weakens perceptual improvement.

The midguard run confirms that weaker regularization mostly returns to the
reuse-gate operating point and still does not dominate it.

Therefore:

```text
Current gate tuning is not the bottleneck.
The bottleneck is condition/control representation quality.
```

## Next Mainline Action

Prioritize:

1. stronger Stage 4 condition adapter / detail-control representation,
2. better sparse control basis or control target,
3. rate-specific adapter/control training after the representation improves.

Do not spend more time on fixed-scale or gate-only sweeps unless a new adapter
or control representation changes the operating point.

## Artifacts

Training:

```text
checkpoints/stage4_cod_lite_condition_gate/20260629_stage5_post_control_condition_gate_detailtarget_long_topk16_smoothguard_ft700_b6.pt
checkpoints/stage4_cod_lite_condition_gate/20260629_stage5_post_control_condition_gate_detailtarget_long_topk16_midguard_ft800_b6.pt
```

Evaluation:

```text
results/stage4_cod_lite_adapter_eval/20260629_stage5_detailtarget_long_topk16_smoothguard_full552/summary.json
results/stage4_cod_lite_adapter_eval/20260629_stage5_detailtarget_long_topk16_midguard_full552/summary.json
```
