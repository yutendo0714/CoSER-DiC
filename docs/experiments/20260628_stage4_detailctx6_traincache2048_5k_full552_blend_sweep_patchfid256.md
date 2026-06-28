# 20260628_stage4_detailctx6_traincache2048_5k_full552_blend_sweep_patchfid256

Date: 2026-06-28T21:15:55

## Summary

| alpha | bpp | PSNR | MS-SSIM | LPIPS-Alex | DISTS | patch-FID |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000 | 0.013999 | 21.9951 | 0.7348 | 0.5758 | 0.3536 | 146.5134 |
| 0.050 | 0.013999 | 22.0226 | 0.7356 | 0.5747 | 0.3550 | 143.4517 |
| 0.100 | 0.013999 | 22.0424 | 0.7361 | 0.5713 | 0.3556 | 138.3503 |
| 0.150 | 0.013999 | 22.0544 | 0.7364 | 0.5658 | 0.3554 | 132.7072 |
| 0.200 | 0.013999 | 22.0586 | 0.7365 | 0.5588 | 0.3547 | 126.1757 |
| 0.250 | 0.013999 | 22.0548 | 0.7364 | 0.5508 | 0.3534 | 119.2942 |
| 0.300 | 0.013999 | 22.0432 | 0.7360 | 0.5421 | 0.3518 | 112.7629 |
| 0.400 | 0.013999 | 21.9964 | 0.7346 | 0.5243 | 0.3479 | 100.7237 |
| 0.500 | 0.013999 | 21.9196 | 0.7322 | 0.5073 | 0.3431 | 90.6889 |
| 0.750 | 0.013999 | 21.6102 | 0.7221 | 0.4717 | 0.3286 | 73.8386 |
| 1.000 | 0.013999 | 21.1677 | 0.7064 | 0.4487 | 0.3127 | 65.1649 |

Payload policy: deterministic decoder-side blend; no extra actual_payload_bpp.

Per-split caution: alpha 0.30 is clean on the full552 aggregate but not on every
split. Kodak24 DISTS changes from 0.3732 to 0.3752, and DIV2K MS-SSIM changes
from 0.6927 to 0.6924. Treat alpha 0.25/0.30 as Stage-4 stability anchors, not
final paper operating points.

Artifacts:

- summary: `results/stage4_blend_sweep/20260628_stage4_detailctx6_traincache2048_5k_full552_blend_sweep_patchfid256/summary.json`
- per-image: `results/stage4_blend_sweep/20260628_stage4_detailctx6_traincache2048_5k_full552_blend_sweep_patchfid256/per_image_metrics.jsonl`
