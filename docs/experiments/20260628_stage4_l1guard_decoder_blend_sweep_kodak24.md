# 20260628_stage4_l1guard_decoder_blend_sweep_kodak24

Date: 2026-06-28T16:58:13

## Summary

| alpha | bpp | PSNR | MS-SSIM | LPIPS-Alex | DISTS | patch-FID |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000 | 0.014121 | 21.6672 | 0.7225 | 0.6366 | 0.3732 | 216.7101 |
| 0.100 | 0.014121 | 21.6815 | 0.7240 | 0.6283 | 0.3745 | 199.7856 |
| 0.200 | 0.014121 | 21.6628 | 0.7241 | 0.6063 | 0.3715 | 181.7010 |
| 0.300 | 0.014121 | 21.6116 | 0.7228 | 0.5807 | 0.3665 | 166.1215 |
| 0.400 | 0.014121 | 21.5292 | 0.7202 | 0.5582 | 0.3596 | 155.1915 |
| 0.500 | 0.014121 | 21.4174 | 0.7163 | 0.5414 | 0.3520 | 146.2623 |
| 0.750 | 0.014121 | 21.0266 | 0.7019 | 0.5137 | 0.3321 | 127.7329 |
| 1.000 | 0.014121 | 20.5132 | 0.6820 | 0.4964 | 0.3151 | 117.5225 |

Payload policy: deterministic decoder-side blend; no extra actual_payload_bpp.

Artifacts:

- summary: `results/stage4_blend_sweep/20260628_stage4_l1guard_decoder_blend_sweep_kodak24/summary.json`
- per-image: `results/stage4_blend_sweep/20260628_stage4_l1guard_decoder_blend_sweep_kodak24/per_image_metrics.jsonl`
