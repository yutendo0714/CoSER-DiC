# 20260629_stage4_fast8192_ft1500_patchfid_kid

Date: 2026-06-29T01:40:00

## Purpose

Evaluate the 8192-image-cache Stage 4 detail-aware CoD-Lite adapter with the
GenCodec-style patch-FID/KID protocol.

The evaluated checkpoint is:

```text
checkpoints/stage4_cod_lite_adapter/
20260629_stage4_detailaware_ft600_fast8192_ft1500_b4.pt
```

The saved full552 reconstructions are:

```text
results/stage4_cod_lite_adapter_eval/
20260629_stage4_detailaware_ft600_fast8192_ft1500_b4_full552_save_eval/
```

## Protocol

```text
strict split:
  Kodak24
  CLIC2020 test428
  DIV2K val100

FID/KID backend:
  torch-fidelity

patch protocol:
  Kodak: 64x64, fid_patch_num=2
  CLIC2020 test: 256x256, fid_patch_num=2
  DIV2K val: 256x256, fid_patch_num=2
```

Patch extraction is used only for FID/KID. PSNR, MS-SSIM, LPIPS, and DISTS are
full-image metrics from the Stage 4 full552 evaluation.

## Results

| dataset | images | patches | patch-FID | patch-KID mean | patch-KID std |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kodak24 | 24 | 2712 | 102.8851 | 0.038774 | 0.001541 |
| CLIC2020 test428 | 428 | 2140 | 66.7065 | 0.010536 | 0.000491 |
| DIV2K val100 | 100 | 500 | 167.7346 | 0.019636 | 0.000000 |

## Comparison

| run | Kodak FID | CLIC FID | DIV2K FID |
| --- | ---: | ---: | ---: |
| bpp0.0156 ft600 | 101.5062 | 67.3085 | 169.1644 |
| bpp0.0312 transfer | 104.7621 | 87.1151 | 182.9174 |
| fast8192 ft1500 | 102.8851 | 66.7065 | 167.7346 |

## Interpretation

The larger clean train cache improved CLIC2020 and DIV2K distribution metrics
and slightly improved full552 LPIPS / condition recovery, but it regressed the
Kodak patch-FID/KID relative to ft600.

This is a mainline signal, not a Stage 5 result:

```text
promote fast8192 ft1500 as the current LPIPS/condition-recovery raw anchor
do not claim official CoD-Lite superiority
require Kodak non-regression for the next promoted adapter
```

## Artifacts

```text
results/analysis/image_distribution_metrics/
20260629_stage4_fast8192_ft1500_kodak_gencodec_patch64_n2_fid_kid.json

results/analysis/image_distribution_metrics/
20260629_stage4_fast8192_ft1500_clic2020_test_gencodec_patch256_n2_fid_kid.json

results/analysis/image_distribution_metrics/
20260629_stage4_fast8192_ft1500_div2k_val_gencodec_patch256_n2_fid_kid.json
```
