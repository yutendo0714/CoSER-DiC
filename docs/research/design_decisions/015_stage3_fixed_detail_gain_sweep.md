# 015 Stage 3 Fixed Detail Gain Sweep

Date: 2026-06-27 JST

## Decision

Keep `detail_gain=1.0` as the neutral Stage 3 quality anchor for now, but keep
`detail_gain=0.9` and `detail_gain=1.1` as fixed decoder-side perception
presets.

These gains do not change the entropy-coded semantic/detail payload. They are
valid as separate fixed codec configurations, not as per-image choices. If a
future encoder selects gain per image, that control decision must be transmitted
and counted.

## Implementation

Added to:

```text
scripts/eval_stage3_uniform_residual_bitstream.py
```

New option:

```bash
--detail-gain FLOAT
```

The decoded residual grid is multiplied by this fixed gain before upsampling and
adding to the semantic reconstruction. The bitstream payload is unchanged. For
`detail_gain != 1.0`, the debug header `entropy_model_version` records a gain
tag. For `detail_gain=1.0`, the version string stays compatible with previous
default Stage 3 runs.

## Protocol

```text
CoSER common LIC
Kodak 24
CLIC Professional Validation 41
DIV2K validation 100, filenames 0801-0900
total: 165 images
crop: 256x256 center crop
actual_payload_bpp: unchanged at 0.018368 for all perceptual sweep points
```

Analysis artifact:

```text
results/analysis/stage3_gain_sweep/20260627_stage3_b5_coser_common_gain_sweep.json
```

## Perceptual Sweep

Baseline:

```text
gain 1.0
run: 20260627_stage3_b5_semposleft_g4_coser_common_lic_perceptual
```

Additional runs:

```text
gain 0.50: 20260627_stage3_b5_semposleft_g4_coser_common_lic_gain050_perceptual
gain 0.75: 20260627_stage3_b5_semposleft_g4_coser_common_lic_gain075_perceptual
gain 0.90: 20260627_stage3_b5_semposleft_g4_coser_common_lic_gain090_perceptual
gain 1.10: 20260627_stage3_b5_semposleft_g4_coser_common_lic_gain110_perceptual
gain 1.25: 20260627_stage3_b5_semposleft_g4_coser_common_lic_gain125_perceptual
```

Summary:

| gain | PSNR | LPIPS | DISTS | LPIPS wins vs 1.0 | DISTS wins vs 1.0 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 21.237605 | 0.694759 | 0.412048 | 84 / 165 | 33 / 165 |
| 0.75 | 21.341496 | 0.694121 | 0.410816 | 107 / 165 | 38 / 165 |
| 0.90 | 21.378554 | 0.694390 | 0.410130 | 117 / 165 | 39 / 165 |
| 1.00 | 21.389952 | 0.694812 | 0.409673 | reference | reference |
| 1.10 | 21.389965 | 0.695422 | 0.409266 | 37 / 165 | 122 / 165 |
| 1.25 | 21.370338 | 0.696642 | 0.408723 | 30 / 165 | 120 / 165 |

Interpretation:

- Lower gains reduce perceptual overshoot and improve LPIPS average.
- Higher gains improve DISTS, and `gain=1.1` preserves PSNR/MS-SSIM almost
  exactly while improving DISTS.
- There is no single fixed gain that dominates PSNR, LPIPS, DISTS, and
  distribution metrics.

## Distribution Metrics

Reconstruction export runs:

```text
gain 0.90: 20260627_stage3_b5_semposleft_g4_coser_common_lic_gain090_recon_export
gain 1.10: 20260627_stage3_b5_semposleft_g4_coser_common_lic_gain110_recon_export
```

Image-level FID/KID:

| gain | FID | KID mean |
| ---: | ---: | ---: |
| 1.00 | 285.503783 | 0.084817 |
| 0.90 | 286.296962 | 0.084582 |
| 1.10 | 284.952160 | 0.084799 |

Patch-FID/KID, 128x128 non-overlapping patches:

| gain | patch-FID | patch-KID mean |
| ---: | ---: | ---: |
| 1.00 | 226.920191 | 0.104232 |
| 0.90 | 227.143411 | 0.104060 |
| 1.10 | 227.114914 | 0.104756 |

Distribution interpretation:

- `gain=1.1` slightly improves image-level FID/KID but worsens patch-level
  KID, so it is not a universal distribution-metric win.
- `gain=0.9` improves image KID and patch KID but worsens FID.
- Keep FID/KID as secondary evidence; do not promote a gain solely from one
  distribution metric.

## Research Takeaway

The fixed residual grid is doing useful low-frequency/detail correction, but the
optimal residual magnitude depends on the perceptual metric. This supports a
future Stage 4/5 design with an explicit rate-perception control path, but the
control must be fixed by codec configuration or counted as transmitted side
information.

Near-term recommendation:

```text
neutral quality anchor: detail_gain = 1.0
LPIPS-leaning preset: detail_gain = 0.9
DISTS/image-FID-leaning preset: detail_gain = 1.1
```

Do not claim a new SOTA result from this sweep. It is a useful no-extra-payload
calibration knob and a diagnostic for the Stage 4 generative decoder.
