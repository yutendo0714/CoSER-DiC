# 059 Stage 4 DISTS Adapter and Counted-Control Guardrails

Date: 2026-06-29 JST

## Decision

Promote `20260629_stage4_detailfilm_stats_hf_dists_ft1000_b8` as the current
balanced no-extra-bit Stage 4 adapter anchor, but do not claim it is a Stage 5
external-baseline result.

Keep the previous sparse counted-control Stage 5 point as the best perceptual
counted-control anchor. The new DISTS adapter plus newly fitted counted-control
basis is useful as a slightly more fidelity-preserving alternative, not as a
strict perceptual replacement.

## Why

The DISTS-loss continuation improves the full552 no-extra-bit adapter without
changing `actual_payload_bpp`:

```text
run:
  20260629_stage4_detailfilm_stats_hf_dists_ft1000_b8_full552_eval

actual_payload_bpp:
  0.013999

metrics:
  PSNR     21.1554
  MS-SSIM   0.7159
  LPIPS     0.4198
  DISTS     0.2875

delta vs 20260629_stage4_detailfilm_stats_hf_ft1200_b12_full552_eval:
  PSNR    +0.0173
  MS-SSIM +0.0007
  LPIPS   -0.0023
  DISTS   -0.0010
```

The improvement is small but consistent enough to use as the next no-extra
adapter base. Its condition L1 is slightly worse, which reinforces the current
policy that condition L1 is only a diagnostic and final decoded metrics decide
promotion.

## Counted-Control Result

A new post-affine G32/S8/K256 basis was fit from the DISTS adapter:

```text
basis:
  outputs/stage5_control_basis/20260629_dists_detailfilm_postaffine_basis_g32s8_k256_train8192_b12/control_basis.pt

coefficient p99:
  0.720179
```

The best low-control-bpp full552 point:

```text
run:
  20260629_stage5_dists_g32basis_full552_affinebasis_ag16_as1_bg32_bs8_k32_b4_topkc256_gr1_br0.25_r0.720179_sc1_huffman_mu_law_mu16_topk_c256_k32_p99_b4_mulaw16

actual_payload_bpp:
  0.015478

metrics:
  PSNR     21.0782
  MS-SSIM   0.7138
  LPIPS     0.3922
  DISTS     0.2727

delta vs no-extra DISTS adapter:
  bpp     +0.001479
  PSNR    -0.0773
  MS-SSIM -0.0020
  LPIPS   -0.0276
  DISTS   -0.0148
```

Compared with the previous stats-HF counted-control topk32 anchor, this new
point is more fidelity preserving but not more perceptual:

```text
delta vs 20260629_stage5_stats_hf_g32basis_full552...topk_c256_k32...:
  bpp     -0.000003
  PSNR    +0.0175
  MS-SSIM +0.0011
  LPIPS   +0.0003
  DISTS   +0.0001
```

Therefore:

```text
best perceptual counted-control anchor:
  20260629_stage5_stats_hf_g32basis_full552...topk_c256_k32...

balanced/fidelity-tilted counted-control alternative:
  20260629_stage5_dists_g32basis_full552...topk_c256_k32...
```

The prefix32+topk32 fixed-bit point is slightly better in LPIPS/DISTS than the
new topk32 point but costs about +0.00096 more bpp, so topk32 remains the
preferred low-rate counted-control point.

## Split Check

For the new topk32 counted-control point, the improvement over the no-extra
DISTS adapter is consistent across the strict CoD 512 splits:

```text
delta vs no-extra DISTS adapter:

Kodak24:
  bpp     +0.001476
  PSNR    -0.1637
  MS-SSIM -0.0044
  LPIPS   -0.0349
  DISTS   -0.0222

CLIC2020 test428:
  bpp     +0.001483
  PSNR    -0.0704
  MS-SSIM -0.0018
  LPIPS   -0.0271
  DISTS   -0.0140

DIV2K val100:
  bpp     +0.001466
  PSNR    -0.0858
  MS-SSIM -0.0022
  LPIPS   -0.0281
  DISTS   -0.0166
```

This confirms counted-control still trades fidelity for perception, but does
so consistently rather than only on Kodak or only on CLIC.

## Tooling Guardrails Added

Several small tooling fixes were made because they directly affect paper-metric
trustworthiness:

```text
scripts/eval_stage4_cod_lite_adapter.py:
  full Stage 4 evaluation now requires actual payload bpp when requested.
  Missing per-image payload bpp no longer silently becomes 0.

scripts/fit_stage5_condition_control_basis.py:
  --huffman-quantile now accepts both p99 and 0.99 forms.
  stdout now prints a compact summary instead of full Huffman tables.

scripts/sweep_stage5_counted_control.py:
  default --limit is now 0, matching eval full-manifest behavior.
  Short screening must explicitly pass --limit 64.
```

These are not method contributions, but they prevent misleading bpp or split
labels from entering future comparisons.

## Next

The next mainline move should not be another fixed output blend or a larger
hand sweep of the same control format. The higher-value direction is:

```text
1. Train a stronger condition adapter that directly improves condition recovery
   without needing counted control.
2. Make the detail stream diffusion-friendly by adding a detail-control head
   from the same transmitted detail payload.
3. Build more operating points for curve comparison only after the adapter
   produces a larger perceptual jump.
4. Keep official CoD-Lite / CoD baselines in view; current results still do not
   beat official external checkpoints.
```
