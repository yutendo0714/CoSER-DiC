# 20260628_stage4_lpips020_decoder_blend_sweep_kodak24

Date: 2026-06-28T17:01:40

## Summary

| alpha | bpp | PSNR | MS-SSIM | LPIPS-Alex | DISTS | patch-FID |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000 | 0.014121 | 21.6672 | 0.7225 | 0.6366 | 0.3732 | 216.7101 |
| 0.050 | 0.014121 | 21.6779 | 0.7236 | 0.6322 | 0.3739 | 205.8748 |
| 0.100 | 0.014121 | 21.6783 | 0.7242 | 0.6203 | 0.3725 | 193.2827 |
| 0.200 | 0.014121 | 21.6483 | 0.7240 | 0.5877 | 0.3661 | 171.2066 |
| 0.300 | 0.014121 | 21.5782 | 0.7220 | 0.5576 | 0.3563 | 155.9178 |
| 0.400 | 0.014121 | 21.4702 | 0.7184 | 0.5355 | 0.3453 | 144.6115 |
| 0.500 | 0.014121 | 21.3270 | 0.7132 | 0.5181 | 0.3349 | 134.4666 |
| 0.750 | 0.014121 | 20.8411 | 0.6949 | 0.4880 | 0.3117 | 116.7533 |
| 1.000 | 0.014121 | 20.2233 | 0.6708 | 0.4753 | 0.2947 | 109.4019 |

Payload policy: deterministic decoder-side blend; no extra actual_payload_bpp.

Artifacts:

- summary: `results/stage4_blend_sweep/20260628_stage4_lpips020_decoder_blend_sweep_kodak24/summary.json`
- per-image: `results/stage4_blend_sweep/20260628_stage4_lpips020_decoder_blend_sweep_kodak24/per_image_metrics.jsonl`
