# Stage 4 Detail-Aware Condition Adapter

Date: 2026-06-28

## Decision

Promote the next Stage 4 mainline experiment from a plain semantic-latent
adapter to a detail-aware CoSER-to-CoD-Lite condition adapter.

This is not an RGB postprocess and not a fixed output blend. It keeps the same
CoSER semantic/detail payload and changes how decoded CoSER information is used
to recover the CoD-Lite condition tensor.

Core path:

```text
semantic VQ stream + detail residual stream
-> decoded x_sem, x_aux, semantic_latent, residual_grid_hat, detail_codes
-> CoSER-owned condition adapter with detail-space residual processing
-> frozen official CoD-Lite one-step backbone
```

## Implementation

Changed files:

```text
src/coserdic/models/gencodec_backbone.py
scripts/train_stage4_cod_lite_adapter.py
tests/test_gencodec_backbone.py
```

Main additions:

```text
CoSERToCoDLiteConditionPyramidAdapterConfig.num_detail_blocks
```

When `num_detail_blocks > 0`, the existing transmitted detail payload
(`residual_grid_hat` + normalized `detail_codes`) is projected to condition
resolution and passed through detail-specific residual blocks before fusion.

The detail residual blocks are identity-initialized:

```text
new detail path starts from the old adapter behavior
new capacity is learned instead of randomly perturbing the old condition map
```

Training support:

```text
--num-detail-blocks
--num-image-blocks
--num-condition-blocks
--num-fusion-blocks
--init-nonstrict
```

`--init-nonstrict` lets a new detail-aware adapter inherit an older
stats-match checkpoint while initializing only the added detail blocks.

Infrastructure fix:

```text
CoDLiteOneStepBackbone.forward now supports batch > 1 by safely looping over
samples before calling the official CoD-Lite inference path.
```

Reason:

```text
official CoD-Lite one-step inference is effectively batch-1 in its modulation
path; direct batch=2 failed with a shape error. The wrapper now absorbs this
official-implementation constraint without changing CoSER semantics.
```

Follow-up shape fix:

```text
CoDLiteOneStepBackbone.condition_size now uses the true native condition tensor
shape, not only the official compressed index-grid size. This is needed for
up2x rate checkpoints such as CoD_Lite_bpp_0_0078.pt.
```

## Training Run

First detail-aware run:

```text
run:
  20260628_stage4_detailaware_adapter_idstart_ft600_b2

init:
  checkpoints/stage4_cod_lite_adapter/
  20260628_stage4_cod_lite_pyramid_sem256_detailctx6_statsmatch_ft2k_b8.pt

checkpoint:
  checkpoints/stage4_cod_lite_adapter/
  20260628_stage4_detailaware_adapter_idstart_ft600_b2.pt

train cache:
  20260628_stage3_gencodec512_traincache_clic_oi_2048_seed20260628

batch:
  batch_size = 2
  grad_accum_steps = 1

adapter:
  pyramid
  semantic_channels = 256
  detail_context = residual_grid_codes
  hidden_channels = 256
  num_detail_blocks = 3

payload:
  unchanged Stage 3 actual_payload_bpp
```

Training summary:

```text
condition_l1_mean:
  0.379980

base_condition_l1_mean:
  0.517428

condition_l1_delta_vs_base_mean:
  -0.137448

lpips_mean:
  0.391246

stage4_psnr_mean:
  21.3158
```

Longer continuation:

```text
run:
  20260628_stage4_detailaware_adapter_idstart_ft3000_b2

init:
  checkpoints/stage4_cod_lite_adapter/
  20260628_stage4_detailaware_adapter_idstart_ft600_b2.pt

checkpoint:
  checkpoints/stage4_cod_lite_adapter/
  20260628_stage4_detailaware_adapter_idstart_ft3000_b2.pt

max_steps:
  3000

lr:
  1.0e-5
```

Training summary:

```text
condition_l1_mean:
  0.377124

base_condition_l1_mean:
  0.517541

condition_l1_delta_vs_base_mean:
  -0.140417

lpips_mean:
  0.386223

stage4_psnr_mean:
  21.3913
```

## Limit64 Evaluation

Same first 64 images from the strict 512 full552 split.

```text
actual_payload_bpp:
  0.013654
```

| run | PSNR | MS-SSIM | LPIPS | DISTS | condition_l1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage 3 anchor | 22.3652 | 0.7568 | 0.5450 | 0.3497 | n/a |
| stats-match raw | 21.5103 | 0.7364 | 0.4213 | 0.3018 | 0.4177 |
| guarded ft300 raw | 21.6261 | 0.7398 | 0.4236 | 0.3037 | 0.4179 |
| detail-aware ft600 raw | 21.6722 | 0.7392 | 0.4142 | 0.2986 | 0.4170 |
| detail-aware ft3000 raw | 21.7478 | 0.7418 | 0.4151 | 0.2992 | 0.4184 |

Interpretation:

```text
detail-aware adapter improves over the previous raw adapter in both fidelity
and perceptual metrics on the same subset and unchanged actual payload.
Continuing to ft3000 improves fidelity further, but slightly gives back
LPIPS/DISTS relative to ft600 on this subset.
```

## Full552 Evaluation

Strict 512 GenCodec-style split:

```text
Kodak24 + CLIC2020 test428 + DIV2K val100 = 552 images
actual_payload_bpp = 0.013999
```

| run | PSNR | MS-SSIM | LPIPS | DISTS | condition_l1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage 3 anchor | 21.9951 | 0.7348 | 0.5758 | 0.3536 | n/a |
| stats-match raw | 21.1157 | 0.7126 | 0.4428 | 0.3040 | 0.4165 |
| condition-gate fidelity probe | 21.3169 | 0.7161 | 0.5255 | 0.3437 | 0.4621 |
| detail-aware ft600 raw | 21.2840 | 0.7155 | 0.4358 | 0.3011 | 0.4162 |
| detail-aware ft3000 raw | 21.3638 | 0.7182 | 0.4384 | 0.3023 | 0.4167 |

Differences vs previous stats-match raw:

```text
ft600:
PSNR:    +0.1683 dB
MS-SSIM: +0.0029
LPIPS:   -0.0069
DISTS:   -0.0029

ft3000:
PSNR:    +0.2481 dB
MS-SSIM: +0.0056
LPIPS:   -0.0044
DISTS:   -0.0017
```

This is a real mainline improvement, not a post-hoc RGB blend.

ft600 and ft3000 now serve different diagnostic roles:

```text
ft600:
  perceptual-leaning raw adapter anchor

ft3000:
  bpp0.0156 continuation / fidelity-recovery check
```

The non-monotonic LPIPS/DISTS behavior means that simply training longer is not
the main answer. The next improvement should come from better condition-space
control, rate-specific backbone initialization, and larger clean training data,
not from extending the same loss indefinitely.

## Mainline Condition-Gate Follow-up

The first condition-space gate probe on top of `ft600` was intentionally kept
off the RGB output path:

```text
train:
  20260628_stage4_detailaware_ft600_condition_gate_probe600_b2

checkpoint:
  checkpoints/stage4_cod_lite_condition_gate/
  20260628_stage4_detailaware_ft600_condition_gate_probe600_b2.pt
```

Limit64 result:

| run | PSNR | MS-SSIM | LPIPS | DISTS | condition_gate_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| ft600 raw | 21.6722 | 0.7392 | 0.4142 | 0.2986 | 0.0000 |
| ft600 condition gate | 21.6889 | 0.7397 | 0.4177 | 0.3011 | 0.9645 |

Judgment:

```text
not promoted
the gate mostly collapsed toward the raw adapter branch
it slightly improves fidelity but worsens LPIPS/DISTS on the same subset
next gate needs stronger anti-collapse constraints and final-image promotion
criteria
```

## Patch-FID / KID

Saved-image artifact:

```text
results/stage4_cod_lite_adapter_eval/
20260628_stage4_detailaware_adapter_idstart_ft600_raw_full552_save_eval/
```

Patch protocol:

```text
backend:
  torch-fidelity

Kodak:
  patch_size = 64
  fid_patch_num = 2

CLIC2020 test / DIV2K val:
  patch_size = 256
  fid_patch_num = 2
```

| dataset | images | patches | patch-FID | patch-KID mean |
| --- | ---: | ---: | ---: | ---: |
| Kodak | 24 | 2712 | 101.5062 | 0.037533 |
| CLIC2020 test | 428 | 2140 | 67.3085 | 0.011038 |
| DIV2K val | 100 | 500 | 169.1644 | 0.021140 |

Artifacts:

```text
results/analysis/image_distribution_metrics/
20260628_stage4_detailaware_ft600_kodak_gencodec_patch64_n2_fid_kid.json

results/analysis/image_distribution_metrics/
20260628_stage4_detailaware_ft600_clic2020_test_gencodec_patch256_n2_fid_kid.json

results/analysis/image_distribution_metrics/
20260628_stage4_detailaware_ft600_div2k_val_gencodec_patch256_n2_fid_kid.json
```

## Current Judgment

This result should become the current Stage 4 mainline baseline:

```text
best no-extra-bit raw CoD-Lite condition-adapter family so far
same actual_payload_bpp as Stage 3
uses decoded CoSER semantic/detail streams directly
improves previous stats-match raw full552 means
```

Current anchors:

```text
use ft600 when judging perceptual raw-adapter gains
use ft3000 when judging whether more training on the same bpp0.0156 prior helps
do not merge these into a Stage 5 claim
```

Rate-specific CoD-Lite backbone probe:

```text
decision:
  docs/research/design_decisions/
  028_stage4_rate_specific_cod_lite_backbone_probe.md

summary:
  bpp0.0078 is not promoted from the first probe
  bpp0.0312 becomes the DISTS/fidelity-leaning raw adapter candidate
  bpp0.0312 does not replace ft600 as the LPIPS/FID anchor
```

But it is still not Stage 5 and not a publishable external-baseline win.

Official CoD-Lite Kodak512 at about the same nominal rate remains far ahead in
LPIPS/DISTS/FID. The correct interpretation is:

```text
The bottleneck is now finer condition recovery and diffusion control,
not whether the CoSER bitstream can drive the CoD-Lite backbone at all.
```

## Next Steps

1. Retrain a condition-space gate on top of the ft600/ft3000 detail-aware anchors.
2. Add split-wise visual failure audit from the saved reconstructions.
3. Test rate-specific CoD-Lite initialization at the same CoSER payload.
4. Expand the clean train cache before judging adapter ceiling.
5. After adapter/gate saturation, try partial low-LR unfreeze or LoRA.
