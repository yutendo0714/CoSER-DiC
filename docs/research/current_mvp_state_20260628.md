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

## Stage 4 Bootstrap Status

CoD-Lite adapter bootstrap has started:

```text
implementation:
  src/coserdic/models/gencodec_backbone.py
  scripts/train_stage4_cod_lite_adapter.py

decision:
  docs/research/design_decisions/022_stage4_cod_lite_adapter_bootstrap.md

status:
  official CoD-Lite one-step checkpoint loads through a CoSER wrapper
  frozen CoD-Lite backbone can decode from CoSER-produced condition tensors
  native-stage3 base condition is used so zero-init starts from decoded x_aux
  512-crop adapter training runs at batch size 1 on the local GPU
  LPIPS loss path runs without immediate NaNs

not promoted:
  raw adapter outputs do not yet beat external baselines
  visual samples still show generative structure distortion
```

Updated active track:

```text
decision:
  docs/research/design_decisions/023_stage4_semantic_latent_cod_lite_adapter.md

implementation:
  Stage 3 can now export decoder_feature_cache per image
  Stage 4 adapter can consume decoded 256-channel semantic latent tensors
  Stage 4 adapter can optionally consume decoded detail context
    residual_grid_hat
    normalized detail_codes
  condition residuals can be tanh-limited and gradient-clipped
  Stage 4 training supports gradient accumulation for larger effective batch

payload policy:
  semantic_latent is decoded from transmitted CoSER semantic tokens
  residual_grid_hat and detail_codes are decoded from transmitted CoSER detail tokens
  adapter and fixed CoD-Lite weights are decoder-side parameters
  no extra image-specific side information is counted unless introduced later
```

Kodak24 semantic-latent probe:

```text
stage3 feature cache:
  20260628_stage3_gencodec512_kodak24_feature_cache_smoke

train:
  20260628_stage4_cod_lite_pyramid_sem256_tanh075_kodak24_probe1k

eval:
  20260628_stage4_cod_lite_pyramid_sem256_tanh075_kodak24_probe1k_eval

actual_payload_bpp:
  0.014123

Stage 3:
  PSNR / MS-SSIM: 21.6674 / 0.7225
  LPIPS / DISTS: 0.6366 / 0.3732

Stage 4 raw:
  PSNR / MS-SSIM: 20.6458 / 0.6911
  LPIPS / DISTS: 0.4294 / 0.2743
```

Semantic latent input is better than RGB semantic input for the CoD-Lite
condition adapter, but the result still trails official CoD-Lite at comparable
rate. The next mainline run should train the semantic-latent + detail-context
adapter on non-evaluation train-cache exports, then evaluate on the strict
552-image CoD 512 reproduction split.

Strict full552 semantic-latent generalization:

```text
feature cache:
  20260628_stage3_gencodec512_full552_feature_cache

raw eval:
  20260628_stage4_cod_lite_pyramid_sem256_tanh075_kodak24_probe1k_full552_eval

alpha0.2 eval:
  20260628_stage4_cod_lite_pyramid_sem256_tanh075_kodak24_probe1k_full552_alpha020_eval

actual_payload_bpp:
  0.013999

Stage 3:
  PSNR / MS-SSIM: 21.9951 / 0.7348
  LPIPS / DISTS: 0.5758 / 0.3536

Stage 4 raw:
  PSNR / MS-SSIM: 20.5700 / 0.7002
  LPIPS / DISTS: 0.4657 / 0.2970

Stage 4 alpha0.2:
  PSNR / MS-SSIM: 21.9958 / 0.7368
  LPIPS / DISTS: 0.5560 / 0.3513
```

The alpha0.2 point is now the safest full-split Stage 4 internal candidate. It
improves all measured means over Stage 3 at unchanged payload. It still does
not close the official CoD-Lite perceptual baseline gap, so it is not a Stage 5
claim.

Strict full552 train-cache semantic-latent + detail-context update:

```text
train:
  20260628_stage4_cod_lite_pyramid_sem256_detailctx6_tanh075_traincache2048_5k_b8

train cache:
  20260628_stage3_gencodec512_traincache_clic_oi_2048_seed20260628

checkpoint:
  checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_tanh075_traincache2048_5k_b8.pt

eval raw:
  20260628_stage4_cod_lite_pyramid_sem256_detailctx6_tanh075_traincache2048_5k_b8_full552_eval

eval alpha0.2:
  20260628_stage4_cod_lite_pyramid_sem256_detailctx6_tanh075_traincache2048_5k_b8_full552_alpha020_eval

blend sweep:
  20260628_stage4_detailctx6_traincache2048_5k_full552_blend_sweep_patchfid256
```

Condition prediction improved substantially:

```text
base_condition_l1: 0.5371
semantic-latent Kodak24 condition_l1: 0.4711
train-cache detail-context condition_l1: 0.4112
```

Full552 patch-FID256 sweep at unchanged actual_payload_bpp=0.013999:

```text
alpha  PSNR     MS-SSIM  LPIPS    DISTS    patch-FID
0.00   21.9951  0.7348   0.5758   0.3536   146.5134
0.20   22.0586  0.7365   0.5588   0.3547   126.1757
0.25   22.0548  0.7364   0.5508   0.3534   119.2942
0.30   22.0432  0.7360   0.5421   0.3518   112.7629
0.40   21.9964  0.7346   0.5243   0.3479   100.7237
1.00   21.1677  0.7064   0.4487   0.3127    65.1649
```

Updated interpretation:

```text
On aggregate full552, alpha0.25 and alpha0.30 supersede the earlier fixed
alpha0.2 point as the best Stage-4 decoder-side candidates. alpha0.30 improves
PSNR, MS-SSIM, LPIPS, DISTS, and patch-FID over Stage 3 without extra payload.
This is meaningful MVP progress, but it is still not an external-baseline win
over official CoD-Lite.
```

Per-split caution:

```text
alpha0.30 vs Stage 3:
  Kodak24:
    PSNR 21.6674 -> 21.6829
    MS-SSIM 0.7225 -> 0.7230
    LPIPS 0.6366 -> 0.6017
    DISTS 0.3732 -> 0.3752

  CLIC2020 test428:
    PSNR 22.3638 -> 22.4199
    MS-SSIM 0.7454 -> 0.7470
    LPIPS 0.5610 -> 0.5276
    DISTS 0.3499 -> 0.3477

  DIV2K val100:
    PSNR 20.4956 -> 20.5173
    MS-SSIM 0.6927 -> 0.6924
    LPIPS 0.6245 -> 0.5899
    DISTS 0.3648 -> 0.3639
```

So alpha0.25/0.30 are stability anchors, not final paper operating points. The
next Stage-4/5 step should remove these split-specific regressions before
claiming a robust improvement.
```

Oracle adaptive-alpha upper bound:

```text
invalid as a codec result:
  alpha is selected per image using reference metrics

min-LPIPS, constrained by PSNR/MS-SSIM >= Stage 3:
  full552 PSNR / MS-SSIM: 22.0349 / 0.7375
  full552 LPIPS / DISTS: 0.5455 / 0.3503

min-DISTS, constrained by PSNR/MS-SSIM >= Stage 3:
  full552 PSNR / MS-SSIM: 22.0217 / 0.7367
  full552 LPIPS / DISTS: 0.5551 / 0.3474
```

This suggests a decoder-side content gate is worth testing, but only if the
gate is computed deterministically from decoded semantic/detail features and
fixed model state. If transmitted, gate/control bits must be counted in
actual_payload_bpp.

Stage 4 measured snapshot on Kodak24 512:

```text
run:
  20260628_stage4_cod_lite_adapter_lpips005_l1guard_full512_5k_kodak24_eval_save24

actual_payload_bpp:
  0.014121

Stage 3 no postprocess:
  PSNR / MS-SSIM: 21.6672 / 0.7225
  LPIPS / DISTS: 0.6366 / 0.3732
  GenCodec Kodak patch-FID: 216.7101

Stage 4 CoD-Lite adapter lpips005 l1guard:
  PSNR / MS-SSIM: 20.5132 / 0.6822
  LPIPS / DISTS: 0.4964 / 0.3151
  GenCodec Kodak patch-FID: 117.5225
```

This is a real perceptual/distribution improvement over Stage 3 at unchanged
CoSER payload, but it is not yet a large improvement over external pretrained
baselines.

Historical deterministic blend anchor:

```text
run:
  20260628_stage4_cod_lite_adapter_lpips020_alpha010_full552_eval

definition:
  stage4 = 0.9 * stage3 + 0.1 * CoD-Lite-adapter-output
  alpha is fixed decoder-side configuration, not image-specific side info

actual_payload_bpp:
  0.013999

strict 512 split:
  Kodak24 + CLIC2020 test428 + DIV2K val100
  total: 552 images

Stage 3:
  PSNR / MS-SSIM: 21.9951 / 0.7348
  LPIPS / DISTS: 0.5758 / 0.3536

Stage 4 alpha0.1:
  PSNR / MS-SSIM: 22.0147 / 0.7365
  LPIPS / DISTS: 0.5626 / 0.3529

win rates:
  PSNR:    0.7627
  MS-SSIM: 0.9601
  LPIPS:   0.9982
  DISTS:   0.5507
```

This was the first safe internal Stage 4 candidate: it improves Stage 3 on all
measured full-split means without adding transmitted bits. It is now superseded
by the train-cache semantic-latent + detail-context alpha0.25/alpha0.30 points
above. The gain is still not large enough for the Stage 5 claim, and it still
trails official CoD-Lite pretrained baselines in perceptual metrics.

Official CoD-Lite Kodak24 512 baseline curve:

```text
0.003906 bpp: PSNR 19.0304, MS-SSIM 0.5700, LPIPS 0.3390, DISTS 0.2089, FID 51.2190
0.007812 bpp: PSNR 19.7975, MS-SSIM 0.6283, LPIPS 0.2807, DISTS 0.1733, FID 44.6564
0.015625 bpp: PSNR 20.7667, MS-SSIM 0.7090, LPIPS 0.2259, DISTS 0.1402, FID 38.4785
0.031250 bpp: PSNR 21.9853, MS-SSIM 0.7811, LPIPS 0.1614, DISTS 0.1120, FID 31.8072
```

Baseline doc:

```text
docs/research/baselines/cod_lite_official_baseline_20260628.md
```

Research implication:

```text
Do not claim existing-baseline superiority yet.
Use the alpha0.25/alpha0.30 detail-context blend as the current stability anchor.
The next Stage 4 design must learn a content-aware decoder-side gate or
stronger structure-preserving adapter that moves toward the official CoD-Lite
curve without losing the Stage 3 fidelity advantage.
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

Stage 4 adapter bootstrap:
  docs/research/design_decisions/022_stage4_cod_lite_adapter_bootstrap.md

Stage 4 semantic-latent adapter:
  docs/research/design_decisions/023_stage4_semantic_latent_cod_lite_adapter.md

Pretrained inventory:
  docs/research/baselines/pretrained_asset_inventory_20260628.md

Bpp policy:
  docs/research/design_decisions/010_bpp_reporting_policy.md
```
