# 20260628_stage4_detailaware_bpp0312_patchfid_kid

Date: 2026-06-29T00:29:00

## Purpose

Evaluate the rate-specific CoD-Lite bpp0.0312 transfer adapter with the
GenCodec-style patch-FID/KID protocol.

This run uses saved `reference` and `stage4` reconstructions from:

```text
results/stage4_cod_lite_adapter_eval/
20260628_stage4_detailaware_adapter_bpp0312_transfer_ft600_b2_full552_save_eval/
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

Patch extraction is used only for distribution metrics. PSNR, MS-SSIM, LPIPS,
and DISTS remain image-level metrics from the full552 Stage 4 evaluation.

## Results

| dataset | images | patches | patch-FID | patch-KID mean | patch-KID std |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kodak24 | 24 | 2712 | 104.7621 | 0.036518 | 0.001126 |
| CLIC2020 test428 | 428 | 2140 | 87.1151 | 0.025608 | 0.000864 |
| DIV2K val100 | 100 | 500 | 182.9174 | 0.038770 | 0.000000 |

## Interpretation

The bpp0.0312 transfer improves raw full552 PSNR, MS-SSIM, and DISTS compared
with the bpp0.0156 detail-aware adapters, but it regresses LPIPS and
patch-FID/KID relative to the ft600 perceptual anchor.

Therefore:

```text
bpp0.0312 transfer:
  keep as DISTS/fidelity-leaning raw backbone candidate

bpp0.0156 ft600:
  keep as LPIPS/FID-leaning raw backbone candidate
```

This is not a Stage 5 result and not an external CoD-Lite baseline win.

## Artifacts

```text
results/analysis/image_distribution_metrics/
20260628_stage4_detailaware_bpp0312_kodak_gencodec_patch64_n2_fid_kid.json

results/analysis/image_distribution_metrics/
20260628_stage4_detailaware_bpp0312_clic2020_test_gencodec_patch256_n2_fid_kid.json

results/analysis/image_distribution_metrics/
20260628_stage4_detailaware_bpp0312_div2k_val_gencodec_patch256_n2_fid_kid.json
```
