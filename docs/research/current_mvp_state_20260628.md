# Current Core MVP State

Date: 2026-06-28 JST  
Status: Active navigation memo for Stage 4 / Stage 5 work

## Bottom Line

The project is ready to start the Core-MVP Stage 4 diffusion integration from
the valid Stage 1-3 bitstream path.

This does not mean CoSER-DiC has already shown a large improvement over
published external methods. It means the internal codec substrate is now stable
enough to test the MVP's intended diffusion-backbone hypothesis:

```text
decoded semantic/detail actual bitstream
  -> auxiliary reconstruction and decoded feature controls
  -> CoSER adapter
  -> CoD-Lite default backbone or CoD parallel heavy backbone
  -> perceptual reconstruction
```

The ResUNet decoder-refiner experiments are archived diagnostics only. They
should not be used to choose the Stage 4 architecture.

## Active Stage 1-3 Anchor

Use the batch-16 rate-prior Stage 1 branch as the current internal
rate-perception anchor:

```text
stage1_checkpoint:
  checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt

stage2_crop512_token_prior:
  checkpoints/stage2_token_prior/20260628_stage2_token_prior_crop512_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_amp_es.pt

stage2_crop512_topk2048_entropy_prior:
  outputs/stage2_learned_entropy/20260628_stage2_learned_topk2048_escape_huffman_fit_32768tokens_crop512_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json

stage3_crop512_detail_prior:
  outputs/stage3_residual_entropy/20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_crop512_8192calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json
```

This path is valid for strict 512-crop GenCodec-style evaluation because it has
a 16x16 semantic context, a matching 16x16 residual prior, decoder-prefix-safe
top-k entropy coding, and exact semantic/detail roundtrip.

## Strict 512 Stage 3 Reference Result

Canonical strict 512 GenCodec-split result:

```text
dataset:
  Kodak 24
  CLIC2020 test 428
  DIV2K val 100
  total: 552 images
  crop: 512 x 512 center crop

run:
  20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_gencodec512_pp_unsharp200_perceptual_prefixsafe

actual_payload_bpp / paper_bpp:
  0.013999

semantic/detail bpp:
  0.008997 / 0.005002

roundtrip:
  semantic=true
  detail=true
  failure_count=0

PSNR / MS-SSIM:
  21.989628 / 0.734834

LPIPS / DISTS:
  0.576842 / 0.353785
```

Do not use the earlier teacher-forced top-k 512 run that had semantic
roundtrip failures.

## Stage 3 Operating Points

For Stage 4 conditioning, prefer the actual decoded fields before cosmetic
postprocess as the clean auxiliary input. Fixed gain/unsharp settings are useful
as Stage 3 reporting operating points, not as the main Stage 4 design axis.

Current Stage 3 reporting interpretation:

```text
no-postprocess:
  safest PSNR / neutral auxiliary reconstruction point

gain1.00 + unsharp0.20:
  DISTS-first Stage 3 point

gain0.625 + unsharp0.15:
  balanced LPIPS/FID point

gain0.50 + unsharp0.15:
  strongest checked LPIPS and CoD-style patch-FID point
  sacrifices PSNR and DISTS
```

These settings are fixed decoder-side operating choices. They do not add
image-specific payload bpp. If a future method transmits image-specific gain,
mask, prompt, seed, map, or control information, it must be counted in
`actual_payload_bpp`.

## Refiner Diagnostic Status

The ResUNet-style decoder refiner is not part of the active MVP Stage 4 path.

Keep from it:

```text
validated:
  actual-bpp-preserving decoder-side evaluation scaffold
  conditioning I/O smoke tests
  need for full Kodak / CLIC2020 / DIV2K distribution gates

not validated:
  CoD or CoD-Lite diffusion behavior
  adapter design for official GenCodec backbones
  ability to beat CoD / CoD-Lite baselines
```

Therefore, do not infer that a ResUNet gain or failure will transfer to CoD or
CoD-Lite.

## Active Stage 4 Plan

Default track:

```text
CoD-Lite pretrained backbone:
  external/pretrained/CoD_Lite/CoD_Lite_pretrain.pt
```

Parallel heavy track:

```text
CoD checkpoints:
  external/pretrained/CoD/cod/CoD_pixel_vpred.pt
  external/pretrained/CoD/cod/CoD_latent_vpred.pt
  external/pretrained/CoD/cod/CoD_latent_vpred_64bits.pt
  external/pretrained/CoD/finetuned_one_step_cod/bpp_0_0039.pt
  external/pretrained/CoD/finetuned_one_step_cod/bpp_0_0312.pt
  external/pretrained/CoD/finetuned_one_step_cod/bpp_0_1250.pt
```

The implementation should wrap the official GenCodec/CoD-Lite backbones as
fixed or low-lr decoder components, then train a CoSER-owned conditioning
adapter from scratch. The adapter must consume only decoded information unless
any extra image-specific side information is explicitly entropy-coded and
counted.

First Stage 4 comparisons:

```text
Stage 3 x_aux only
CoD-Lite official rate checkpoint baseline
CoD official / one-step baseline where reproducible
CoSER + frozen CoD-Lite backbone + adapter
CoSER + frozen CoD backbone + adapter
```

Promotion gate:

```text
actual_payload_bpp remains Stage 3 semantic/detail payload bpp
roundtrip failure_count = 0
LPIPS / DISTS improve over Stage 3 at the same payload
FID/KID do not regress under the selected CoD/CoD-Lite protocol
visual failures are audited, especially structured high-frequency content
```

## Stage 5 Definition

Stage 5 starts only after a Stage 4 diffusion path works. It is not simply
"train longer".

Stage 5 means joint Core MVP finetuning:

```text
semantic encoder
detail residual encoder
entropy priors
auxiliary decoder
diffusion adapter
optional very-low-lr diffusion backbone
```

Required ablations before a Stage 5 claim:

```text
semantic only
semantic + detail, no diffusion
diffusion without detail
diffusion without semantic conditioning
CoD-Lite vs CoD backbone
actual_payload_bpp and estimated_bpp reported separately
```

## Main References

```text
Stage 1-3 active anchor:
  docs/research/design_decisions/019_stage1_rateprior_b16_reconnection.md

Refiner archive:
  docs/research/design_decisions/020_stage4_decoder_side_refiner.md

Active Stage 4 policy:
  docs/research/design_decisions/021_stage4_cod_codlite_parallel_backbone_policy.md

Pretrained inventory:
  docs/research/baselines/pretrained_asset_inventory_20260628.md

Bpp policy:
  docs/research/design_decisions/010_bpp_reporting_policy.md
```
