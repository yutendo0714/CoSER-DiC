# Stage 4 Fast Train-Cache and 8192-Image Adapter Scale Decision

Date: 2026-06-29

## Decision

Promote the fast Stage 3 training-cache builder as Stage 4 adapter-training
infrastructure, but keep its payload metrics out of paper-rate reporting.

Promote the 8192-image adapter continuation as the current
LPIPS/condition-recovery raw Stage 4 anchor, not as a Stage 5 result and not as
an external CoD-Lite win.

The mainline next step is not RGB postprocessing. It is stronger CoSER-to-CoD-
Lite condition recovery with better small-split generalization guards.

## Fast Training Cache

New script:

```text
scripts/build_stage3_training_cache_fast.py
```

Purpose:

```text
build Stage 4 adapter train-cache quickly
save decoded CoSER features needed by the adapter
avoid exact semantic entropy roundtrip during cache construction
```

It reconstructs decoded semantic latent from VQ indices and decodes residual
detail codes, but it deliberately skips entropy-coded semantic stream
roundtrip. Therefore:

```text
use fast cache for adapter training
do not use fast cache for actual_payload_bpp / paper_bpp reporting
use exact compress/decompress evaluation for all paper bpp numbers
```

The exact-vs-fast spot check on the same seed/batch4 subset gave:

| check | value |
| --- | ---: |
| source_match | true |
| max_semantic_index_diff | 0 |
| max_semantic_latent_abs | 0.0 |
| max_residual_grid_hat_abs | 0.0 |
| max_detail_code_diff | 0 |
| max_stage3_png_abs | 0.0039216 |

The PNG difference is one 8-bit quantization step. This is acceptable for a
training cache, while exact bitstream evaluation remains mandatory for paper
metrics.

## 8192-Image Cache

Run:

```text
20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4
```

Inputs:

```text
CLIC professional train
CLIC mobile train
OpenImages train
```

Held out from training:

```text
Kodak24
CLIC2020 test428
DIV2K val100
```

Summary:

| metric | value |
| --- | ---: |
| images | 8192 |
| semantic_only_psnr | 21.5069 |
| stage3_psnr | 21.9801 |
| stage3_ms_ssim | 0.7446 |
| residual_grid_clip_ratio | 0.000049 |

Artifact:

```text
results/bitstreams/stage3_training_cache_fast/
20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4
```

## Adapter Scale Run

Run:

```text
20260629_stage4_detailaware_ft600_fast8192_ft1500_b4
```

Init:

```text
checkpoints/stage4_cod_lite_adapter/
20260628_stage4_detailaware_adapter_idstart_ft600_b2.pt
```

Main adapter setup:

```text
CoD-Lite bpp0.0156 backbone
decoded semantic latent, 256 channels
decoded residual_grid/detail context
condition residual form:
  pred_cond = base_cond + 0.85 * tanh(adapter_delta)
batch_size = 4
max_steps = 1500
```

Training summary:

| metric | value |
| --- | ---: |
| base_condition_l1 | 0.5173 |
| condition_l1 | 0.3996 |
| condition_l1_delta_vs_base | -0.1177 |
| lpips | 0.4081 |
| stage3_psnr | 21.9510 |
| stage4_psnr | 21.2086 |

## Full552 Image Metrics

Strict 512 GenCodec-style split:

```text
Kodak24 + CLIC2020 test428 + DIV2K val100
```

Main bpp:

```text
actual_payload_bpp / paper_bpp = 0.013999
```

Full552 comparison:

| run | PSNR | MS-SSIM | LPIPS | DISTS | condition_l1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage 3 anchor | 21.9951 | 0.7348 | 0.5758 | 0.3536 | n/a |
| bpp0.0156 ft600 | 21.2840 | 0.7155 | 0.4358 | 0.3011 | 0.4162 |
| bpp0.0156 ft3000 | 21.3638 | 0.7182 | 0.4384 | 0.3023 | 0.4167 |
| bpp0.0312 transfer | 21.4438 | 0.7216 | 0.4402 | 0.2917 | 0.4589 |
| fast8192 ft1500 | 21.3957 | 0.7186 | 0.4342 | 0.3022 | 0.4095 |

Split summary for fast8192 ft1500:

| split | count | PSNR | MS-SSIM | LPIPS | DISTS | condition_l1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Kodak24 | 24 | 21.1184 | 0.7033 | 0.4720 | 0.3228 | 0.4255 |
| CLIC2020 test428 | 428 | 21.7644 | 0.7309 | 0.4224 | 0.2989 | 0.4054 |
| DIV2K val100 | 100 | 19.8844 | 0.6696 | 0.4756 | 0.3111 | 0.4235 |

Interpretation:

```text
fast8192 improves full552 LPIPS and condition_l1 over ft600/ft3000
fast8192 improves PSNR/MS-SSIM over ft600
fast8192 does not beat bpp0.0312 transfer on DISTS or fidelity
Kodak remains a weak generalization split
```

## GenCodec-Style Patch-FID/KID

Protocol:

```text
torch-fidelity backend
Kodak: 64x64 patches, fid_patch_num=2
CLIC2020 / DIV2K: 256x256 patches, fid_patch_num=2
patch metrics are computed per dataset
```

| run | dataset | patches | patch-FID | patch-KID mean |
| --- | --- | ---: | ---: | ---: |
| ft600 | Kodak24 | 2712 | 101.5062 | 0.037533 |
| fast8192 ft1500 | Kodak24 | 2712 | 102.8851 | 0.038774 |
| ft600 | CLIC2020 test428 | 2140 | 67.3085 | 0.011038 |
| fast8192 ft1500 | CLIC2020 test428 | 2140 | 66.7065 | 0.010536 |
| ft600 | DIV2K val100 | 500 | 169.1644 | 0.021140 |
| fast8192 ft1500 | DIV2K val100 | 500 | 167.7346 | 0.019636 |

The bpp0.0312 transfer remains worse on these patch distribution metrics:

| dataset | patch-FID | patch-KID mean |
| --- | ---: | ---: |
| Kodak24 | 104.7621 | 0.036518 |
| CLIC2020 test428 | 87.1151 | 0.025608 |
| DIV2K val100 | 182.9174 | 0.038770 |

## Judgment

The larger clean cache is useful. It improves the main full552 LPIPS anchor and
condition recovery without adding payload bits.

However, this is still not Stage 5:

```text
not jointly trained
not a BD-rate curve
not an official CoD-Lite / CoD win
not enough Kodak robustness
```

The result points to a better next move:

```text
scale training helps CLIC/DIV2K distribution metrics
Kodak needs a generalization guard or more balanced train sampling
condition recovery still dominates the bottleneck
```

## Next Mainline Actions

Immediate:

```text
1. add balanced train sampling or Kodak-like holdout augmentation
2. train a condition-statistics-aware adapter continuation from fast8192 ft1500
3. make adapter selection require Kodak non-regression against ft600
4. add semantic/detail ablation checks on the fast8192 checkpoint
```

Then:

```text
1. try the same 8192-cache scale with bpp0.0312 initialization
2. train a deterministic condition-space gate, not RGB alpha
3. move to 20k-100k clean cache only if Kodak guard improves
4. build multiple actual_payload_bpp operating points before any BD-rate claim
```

The method remains:

```text
CoSER semantic VQ stream
+ CoSER detail residual stream
+ entropy-coded actual payload
+ CoSER-owned condition adapter/gate
+ frozen or lightly adapted CoD/CoD-Lite diffusion backbone
```
