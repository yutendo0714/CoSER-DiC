# 20260628_cod_lite_condition_oracle_kodak24

Date: 2026-06-28T17:15:09

## Summary

| candidate | bpp | cond L1 to ref | PSNR | MS-SSIM | LPIPS-Alex | DISTS | patch-FID |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stage3_native | 0.014121 | 0.5670 | 21.0403 | 0.6939 | 0.6522 | 0.3941 | 202.5268 |
| semantic_native | 0.014121 | 0.5714 | 20.7686 | 0.6902 | 0.6555 | 0.3952 | 201.7102 |
| reference_native | 0.014121 | 0.0000 | 20.7790 | 0.7092 | 0.2259 | 0.1404 | 38.4467 |
| cond_lerp_0.000 | 0.014121 | 0.5670 | 21.0403 | 0.6939 | 0.6522 | 0.3941 | 202.5268 |
| cond_lerp_0.100 | 0.014121 | 0.5103 | 21.1980 | 0.7024 | 0.6228 | 0.3822 | 183.2781 |
| cond_lerp_0.250 | 0.014121 | 0.4252 | 21.4216 | 0.7151 | 0.5623 | 0.3536 | 145.4955 |
| cond_lerp_0.500 | 0.014121 | 0.2835 | 21.6910 | 0.7322 | 0.4171 | 0.2728 | 83.4207 |
| cond_lerp_0.750 | 0.014121 | 0.1418 | 21.5768 | 0.7343 | 0.2776 | 0.1862 | 48.8391 |
| cond_lerp_1.000 | 0.014121 | 0.0000 | 20.7790 | 0.7092 | 0.2259 | 0.1404 | 38.4467 |

Reference-derived condition candidates are upper-bound diagnostics only.
They are not valid CoSER-DiC paper points because the reference condition is not decoder-available.

## Artifacts

- summary: `results/stage4_cod_lite_condition_oracle/20260628_cod_lite_condition_oracle_kodak24/summary.json`
- per-image: `results/stage4_cod_lite_condition_oracle/20260628_cod_lite_condition_oracle_kodak24/per_image_metrics.jsonl`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_171307-xakregfo`
