# 014 Stage 3 Reconstruction Export and Distribution Metrics

Date: 2026-06-27 JST

## Decision

Add reconstruction export and image-distribution metric tooling before moving
to Stage 4. Stage 3 is still not a perceptual/SOTA claim, but the project now
has the artifact path needed for FID/KID, patch-FID/KID, and failure galleries.

## Implementation

Updated:

```text
scripts/eval_stage3_uniform_residual_bitstream.py
```

New options:

```bash
--save-reconstructions
--save-reconstruction-limit N
--save-reconstruction-triptychs
```

When enabled, the evaluator writes crop-aligned images under:

```text
reconstructions/reference/
reconstructions/semantic_only/
reconstructions/stage3/
reconstructions/triptych/        # only with --save-reconstruction-triptychs
reconstructions/manifest.jsonl
```

New analysis utilities:

```text
scripts/compute_image_distribution_metrics.py
scripts/export_worst_case_gallery.py
```

`compute_image_distribution_metrics.py` uses `torch_fidelity` and supports:

```bash
# image-level FID/KID
--reference-dir DIR --candidate-dir DIR

# single-grid diagnostic patch-FID/KID
--patch-size 128 --patch-stride 128

# CoD paper-style patch sampling
--patch-protocol gencodec --gencodec-dataset-filter kodak
--patch-protocol gencodec --gencodec-dataset-filter clic2020_test --gencodec-other-patch-size 128

# CoD-Lite / GLC / DLF-style high-resolution patch sampling
--patch-protocol gencodec --gencodec-dataset-filter clic2020_test --gencodec-other-patch-size 256
--patch-protocol gencodec --gencodec-dataset-filter div2k_val
```

The patch mode writes an explicit patch cache and records the number of images
and patches used. The legacy `--patch-size 128 --patch-stride 128` path is an
internal single-grid diagnostic. CoD/CoD-Lite-style patch FID must be run per
dataset and labeled with the exact patch size. Kodak512 uses 64px patches with
a 32px shifted second grid. CoD paper CLIC2020 512-crop uses 128px patches with
a 64px shifted second grid. CoD-Lite/GLC/DLF-style CLIC or DIV2K settings often
use 256px patches with a 128px shifted second grid. This patch sampling is for
FID/KID only; PSNR, LPIPS, and DISTS remain image-level.

Current backend note: the CoSER utility computes FID/KID with `torch_fidelity`.
GenCodec's external scripts use `torchmetrics.image.fid.FrechetInceptionDistance`.
For paper tables, label both the patch sampling protocol and backend.

## Smoke Export

Run:

```text
20260627_stage3_reconstruction_export_smoke_kodak4
wandb/offline-run-20260627_164751-ll3rpa3b
```

Artifacts:

```text
results/bitstreams/stage3_uniform_residual/20260627_stage3_reconstruction_export_smoke_kodak4/summary.json
results/bitstreams/stage3_uniform_residual/20260627_stage3_reconstruction_export_smoke_kodak4/reconstructions/manifest.jsonl
results/analysis/stage3_visual_galleries/20260627_stage3_recon_export_smoke_kodak4_lpips_regressions.md
```

Smoke metrics:

```text
num_images: 4
actual_payload_bpp: 0.017548
PSNR delta vs semantic-only: +0.635493 dB
LPIPS delta vs semantic-only: -0.011340
DISTS delta vs semantic-only: -0.008346
roundtrip failures: 0
```

Patch-FID/KID plumbing check on the 4-image smoke used 16 patches:

```text
patch size / stride: 128 / 128
FID: 395.507870
KID mean: 0.132964
```

This value is only a wiring check and is not used as a research claim.

## CoSER Common LIC Export

Run:

```text
20260627_stage3_b5_semposleft_g4_coser_common_lic_recon_export
wandb/offline-run-20260627_164956-y0pxqvuk
```

Protocol:

```text
Kodak 24
CLIC Professional Validation 41
DIV2K validation 100, filenames 0801-0900
total: 165 images
crop: 256x256 center crop
bpp: actual_payload_bpp
```

Artifacts:

```text
results/bitstreams/stage3_uniform_residual/20260627_stage3_b5_semposleft_g4_coser_common_lic_recon_export/summary.json
results/bitstreams/stage3_uniform_residual/20260627_stage3_b5_semposleft_g4_coser_common_lic_recon_export/reconstructions/manifest.jsonl
results/bitstreams/stage3_uniform_residual/20260627_stage3_b5_semposleft_g4_coser_common_lic_recon_export/reconstructions/reference/
results/bitstreams/stage3_uniform_residual/20260627_stage3_b5_semposleft_g4_coser_common_lic_recon_export/reconstructions/semantic_only/
results/bitstreams/stage3_uniform_residual/20260627_stage3_b5_semposleft_g4_coser_common_lic_recon_export/reconstructions/stage3/
```

Core metrics:

```text
actual_payload_bpp: 0.018368
semantic_payload_bpp: 0.010516
detail_payload_bpp: 0.007852
debug_full_stream_bpp: 0.023373
PSNR: 21.389952
PSNR delta vs semantic-only: +0.471572 dB
MS-SSIM delta vs semantic-only: +0.004616
roundtrip failures: 0
reconstruction_count: 165
```

These match the earlier perceptual run, except LPIPS/DISTS were not recomputed
in this export run.

## Distribution Metrics

Reference is the crop-aligned original image from the evaluator.

Image-level FID/KID, 165 images:

```text
semantic-only:
  FID: 290.559459
  KID mean: 0.087668

stage3:
  FID: 285.503783
  KID mean: 0.084817

stage3 improvement:
  FID: -5.055676
  KID mean: -0.002850
```

Patch-FID/KID, 128x128 non-overlapping patches, 660 patches
(`torch_fidelity`, internal diagnostic; not CoD/CoD-Lite dataset-level patch FID):

```text
semantic-only:
  FID: 229.132822
  KID mean: 0.104425

stage3:
  FID: 226.920191
  KID mean: 0.104232

stage3 improvement:
  FID: -2.212632
  KID mean: -0.000193
```

Interpretation:

- Stage 3 residual detail improves FID/KID versus semantic-only in both image
  and patch modes.
- The improvement is modest, consistent with the LPIPS/DISTS deltas from
  Decision 013.
- The absolute FID values remain high. At about 0.018 bpp, Stage 3 is a stable
  low-rate bitstream baseline, not a perceptual-compression SOTA result.
- Stage 4 should target the perceptual gap, while preserving the current
  semantic/detail stream split and actual-payload-bpp accounting.

## Failure Galleries

Worst-case galleries were generated by merging the perceptual per-image JSONL
from Decision 013 with the reconstruction-image paths from the export run.

```text
LPIPS regressions:
  results/analysis/stage3_visual_galleries/20260627_stage3_b5_common_lpips_regressions.md

DISTS regressions:
  results/analysis/stage3_visual_galleries/20260627_stage3_b5_common_dists_regressions.md
```

Top-24 dataset skew:

```text
LPIPS regression top-24:
  DIV2K val: 18
  CLIC professional valid: 5
  Kodak: 1

DISTS regression top-24:
  DIV2K val: 10
  CLIC professional valid: 8
  Kodak: 6
```

Worst LPIPS cases:

```text
/dpl/div2k/0897.png  delta +0.022332, bpp 0.022949
/dpl/div2k/0866.png  delta +0.019437, bpp 0.012085
/dpl/div2k/0893.png  delta +0.016567, bpp 0.010864
```

Worst DISTS cases:

```text
/dpl/div2k/0803.png       delta +0.005828, bpp 0.018311
/dpl/kodak/kodim23.png    delta +0.004859, bpp 0.019775
/dpl/kodak/kodim20.png    delta +0.004476, bpp 0.015747
```

Interpretation: LPIPS regressions are concentrated in DIV2K validation. The
next Stage 3 or Stage 4 tuning pass should inspect whether these are caused by
low-frequency residual overshoot, edge/color residual interpolation artifacts,
or semantic-token failures that the coarse residual grid cannot correct.

## Next Gate

Before promoting a Stage 4 generative model:

1. Export the same reconstruction layout for Stage 4.
2. Report image-level FID/KID and, for CoD/CoD-Lite comparisons, dataset-level
   patch FID with explicit patch size, split count, and backend against the
   same crop-aligned or full-resolution reference
   directories.
3. Generate worst-case galleries for LPIPS/DISTS regressions.
4. Keep `actual_payload_bpp` separate from debug/full-file metrics.
