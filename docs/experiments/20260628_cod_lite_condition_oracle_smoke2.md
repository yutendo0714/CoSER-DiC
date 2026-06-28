# 20260628_cod_lite_condition_oracle_smoke2

Date: 2026-06-28T17:12:45

## Summary

| candidate | bpp | cond L1 to ref | PSNR | MS-SSIM | LPIPS-Alex | DISTS | patch-FID |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stage3_native | 0.013596 | 0.5622 | 22.0853 | 0.6525 | 0.7260 | 0.4141 | 304.9316 |
| reference_native | 0.013596 | 0.0000 | 21.7237 | 0.6838 | 0.2341 | 0.1457 | 96.5839 |
| cond_lerp_0.000 | 0.013596 | 0.5622 | 22.0853 | 0.6525 | 0.7260 | 0.4141 | 304.9316 |
| cond_lerp_0.500 | 0.013596 | 0.2811 | 22.5107 | 0.6949 | 0.4660 | 0.2949 | 185.9731 |
| cond_lerp_1.000 | 0.013596 | 0.0000 | 21.7237 | 0.6838 | 0.2341 | 0.1457 | 96.5839 |

Reference-derived condition candidates are upper-bound diagnostics only.
They are not valid CoSER-DiC paper points because the reference condition is not decoder-available.

## Artifacts

- summary: `results/stage4_cod_lite_condition_oracle/20260628_cod_lite_condition_oracle_smoke2/summary.json`
- per-image: `results/stage4_cod_lite_condition_oracle/20260628_cod_lite_condition_oracle_smoke2/per_image_metrics.jsonl`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_171236-0bmp21tx`
