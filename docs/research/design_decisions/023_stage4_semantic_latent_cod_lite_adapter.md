# Stage 4 Semantic-Latent CoD-Lite Adapter

Date: 2026-06-28 JST
Status: Active MVP mainline
Parent: `022_stage4_cod_lite_adapter_bootstrap.md`
Mainline guardrail: `024_stage4_stage5_mainline_research_direction.md`

## Decision

Return Stage 4 to the Core MVP mainline:

```text
CoSER semantic/detail actual bitstream
  -> exact decoder-side Stage 3 reconstruction
  -> fixed official CoD-Lite decoder backbone
  -> CoSER-owned deterministic condition adapter
```

Decoder-side ResUNet/refiner diagnostics are not used as evidence for CoD /
CoD-Lite integration quality. They remain implementation probes only. Promotion
to Stage 4/5 must be based on the official diffusion decoder path.

Fixed alpha blends in this document are diagnostics and stability anchors. They
are not the proposed method. The next mainline work is condition recovery,
deterministic content gating, diffusion-friendly detail features, training
scale, and official baseline curves.

## Payload Policy

The adapter may consume only decoder-available tensors:

```text
decoded Stage 3 reconstruction
decoded semantic-only reconstruction
decoded residual grid
decoded semantic token indices
decoded semantic quantized latent
```

These tensors are derived from the transmitted Stage 3 semantic/detail streams.
They do not add `actual_payload_bpp`.

The following are fixed decoder-side parameters and are not counted:

```text
official CoD-Lite checkpoint
CoSER condition adapter checkpoint
fixed deterministic blend alpha
fixed decoding code/config
```

Any image-specific prompt, mask, seed, rate map, adaptive table, or codebook
update would have to be counted if introduced later.

## Implementation Update

Added decoder feature cache export to Stage 3 evaluation:

```text
script:
  scripts/eval_stage3_uniform_residual_bitstream.py

flag:
  --save-decoder-feature-cache

saved per image:
  semantic_indices
  semantic_latent
  residual_grid_hat
  detail_codes
```

Updated Stage 4 train/eval scripts to load `decoder_feature_cache` when
`--semantic-channels != 3`. This allows the adapter to use the 256-channel
decoded semantic latent instead of only RGB semantic reconstruction.

The train/eval path now also supports optional decoded detail context:

```text
flag:
  --detail-context residual_grid
  --detail-context residual_grid_codes

decoder-available inputs:
  residual_grid_hat: 3 x 16 x 16
  detail_codes:      3 x 16 x 16, normalized from the fixed 4-bit code range
```

This is not extra side information. It is derived from the already transmitted
Stage 3 detail stream and is only a stronger way to expose existing decoded
information to the CoD-Lite condition adapter.

Stability controls added:

```text
pred_cond = base_cond + scale * tanh(adapter_delta)

flags:
  --condition-residual-tanh
  --condition-residual-scale
  --grad-clip-norm
  --init-checkpoint
  --grad-accum-steps
```

## Oracle Diagnosis

Run:

```text
results/stage4_cod_lite_condition_oracle/20260628_cod_lite_condition_oracle_kodak24
```

Key Kodak24 findings:

```text
stage3_native condition:
  LPIPS / DISTS / FID: 0.6522 / 0.3941 / 202.5268

reference_native oracle condition:
  LPIPS / DISTS / FID: 0.2259 / 0.1404 / 38.4467

cond_lerp_0.75 oracle:
  PSNR / MS-SSIM: 21.5768 / 0.7343
  LPIPS / DISTS / FID: 0.2776 / 0.1862 / 48.8391
```

Interpretation:

```text
The fixed CoD-Lite decoder wrapper is correct.
The main bottleneck is predicting reference-grade CoD-Lite condition tensors
from decoder-available CoSER information.
```

## Kodak24 Adapter Probes

RGB semantic input, tanh-limited pyramid adapter:

```text
train:
  20260628_stage4_cod_lite_pyramid_condgrid_tanh075_bs4_probe2k_lr1e4

eval:
  20260628_stage4_cod_lite_pyramid_condgrid_tanh075_probe2k_kodak24_eval

actual_payload_bpp:
  0.014121

Stage 3:
  PSNR / MS-SSIM: 21.6672 / 0.7225
  LPIPS / DISTS: 0.6366 / 0.3732

Stage 4 raw:
  PSNR / MS-SSIM: 20.7415 / 0.6810
  LPIPS / DISTS: 0.4796 / 0.3258
```

Semantic latent 256-channel input, tanh-limited pyramid adapter:

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

The 256-channel semantic latent is materially better than RGB semantic input
for the CoD-Lite condition adapter, especially on LPIPS and DISTS. This is more
aligned with the CoSER novelty axis because it uses the transmitted semantic
code as a decoder-internal representation, not merely as a rendered image.

## Blend Sweep

Run:

```text
20260628_stage4_cod_lite_pyramid_sem256_tanh075_probe1k_blend_kodak24
```

Selected Kodak24 points:

```text
alpha 0.00:
  PSNR / MS-SSIM: 21.6674 / 0.7225
  LPIPS / DISTS / FID: 0.6366 / 0.3732 / 216.7738

alpha 0.15:
  PSNR / MS-SSIM: 21.7068 / 0.7254
  LPIPS / DISTS / FID: 0.6170 / 0.3707 / 193.3113

alpha 0.20:
  PSNR / MS-SSIM: 21.7032 / 0.7256
  LPIPS / DISTS / FID: 0.6038 / 0.3673 / 183.5398

alpha 1.00:
  PSNR / MS-SSIM: 20.6509 / 0.6911
  LPIPS / DISTS / FID: 0.4296 / 0.2743 / 86.5198
```

The safe blend region improves Stage 3 without changing payload, but it is not
yet a large improvement over official CoD-Lite.

## Full 552 Generalization Check

Full 512 reproduction feature cache:

```text
run:
  20260628_stage3_gencodec512_full552_feature_cache

dataset:
  Kodak24 + CLIC2020 test428 + DIV2K val100
  total: 552

actual_payload_bpp:
  0.013999

roundtrip:
  semantic=true
  detail=true
  failure_count=0
```

Kodak24-trained semantic-latent adapter, raw output:

```text
eval:
  20260628_stage4_cod_lite_pyramid_sem256_tanh075_kodak24_probe1k_full552_eval

Stage 3:
  PSNR / MS-SSIM: 21.9951 / 0.7348
  LPIPS / DISTS: 0.5758 / 0.3536

Stage 4 raw:
  PSNR / MS-SSIM: 20.5700 / 0.7002
  LPIPS / DISTS: 0.4657 / 0.2970

condition:
  base_condition_l1: 0.5371
  condition_l1:      0.4711
  delta_vs_base:    -0.0660
```

The adapter generalizes beyond Kodak24 in condition space and perceptual
metrics, but raw output still damages fidelity.

Kodak24-trained semantic-latent adapter, fixed alpha 0.2 decoder-side blend:

```text
eval:
  20260628_stage4_cod_lite_pyramid_sem256_tanh075_kodak24_probe1k_full552_alpha020_eval

Stage 3:
  PSNR / MS-SSIM: 21.9951 / 0.7348
  LPIPS / DISTS: 0.5758 / 0.3536

Stage 4 alpha0.2:
  PSNR / MS-SSIM: 21.9958 / 0.7368
  LPIPS / DISTS: 0.5560 / 0.3513

win rates:
  PSNR:    0.4438
  MS-SSIM: 0.8261
  LPIPS:   0.9964
  DISTS:   0.6322
```

This was the first safe Stage 4 operating point for full552: all mean metrics
improve over Stage 3 without extra transmitted payload. It is now superseded
by the train-cache semantic-latent + detail-context alpha0.25/alpha0.30 points
below. It is still a Stage-4 internal improvement, not a Stage-5 external-
baseline win.

Train-cache semantic-latent + decoder-available detail-context adapter:

```text
train:
  20260628_stage4_cod_lite_pyramid_sem256_detailctx6_tanh075_traincache2048_5k_b8

training data:
  2048 non-evaluation images from CLIC train + OpenImages

inputs:
  semantic_latent: 256ch cached decoder semantic latent
  detail_context: residual_grid_hat + normalized detail_codes, 6ch

checkpoint:
  checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_tanh075_traincache2048_5k_b8.pt

full552 raw eval:
  20260628_stage4_cod_lite_pyramid_sem256_detailctx6_tanh075_traincache2048_5k_b8_full552_eval

full552 alpha0.2 eval:
  20260628_stage4_cod_lite_pyramid_sem256_detailctx6_tanh075_traincache2048_5k_b8_full552_alpha020_eval

full552 blend sweep:
  20260628_stage4_detailctx6_traincache2048_5k_full552_blend_sweep_patchfid256
```

Condition-space result on full552:

```text
base_condition_l1: 0.5371
condition_l1:      0.4112
delta_vs_base:    -0.1259
```

This is a clear condition-prediction improvement over the Kodak24-trained
semantic-latent adapter (`condition_l1=0.4711`). It confirms that decoder-
available detail information is useful for predicting a CoD-Lite condition
tensor without transmitting extra side information.

Full552 alpha sweep, patch-FID256:

```text
actual_payload_bpp: 0.013999

alpha  PSNR     MS-SSIM  LPIPS    DISTS    patch-FID
0.00   21.9951  0.7348   0.5758   0.3536   146.5134
0.20   22.0586  0.7365   0.5588   0.3547   126.1757
0.25   22.0548  0.7364   0.5508   0.3534   119.2942
0.30   22.0432  0.7360   0.5421   0.3518   112.7629
0.40   21.9964  0.7346   0.5243   0.3479   100.7237
0.50   21.9196  0.7322   0.5073   0.3431    90.6889
0.75   21.6102  0.7221   0.4717   0.3286    73.8386
1.00   21.1677  0.7064   0.4487   0.3127    65.1649
```

Updated Stage-4 interpretation:

```text
On aggregate full552, alpha0.25 and alpha0.30 are stronger candidates than the
earlier fixed alpha0.2 point. They improve PSNR, MS-SSIM, LPIPS, and patch-FID
over Stage 3 at unchanged actual_payload_bpp, while alpha0.30 also improves
DISTS. Raw alpha1.0 is useful diagnostically but still sacrifices too much
fidelity.
```

Per-split caution:

```text
Stage 3 / alpha0.30:
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

The aggregate improvement is real but not uniformly clean per split: Kodak DISTS
slightly worsens at alpha0.30, and DIV2K MS-SSIM slightly worsens. Treat
alpha0.25/0.30 as Stage-4 stability anchors, not final paper operating points.

Oracle adaptive-alpha diagnostic:

```text
This diagnostic selects alpha per image using reference metrics, so it is not a
valid codec result. It estimates the headroom for a future decoder-available
content gate.

min-LPIPS with PSNR and MS-SSIM not below Stage 3:
  full552:
    PSNR / MS-SSIM: 22.0349 / 0.7375
    LPIPS / DISTS: 0.5455 / 0.3503

min-DISTS with PSNR and MS-SSIM not below Stage 3:
  full552:
    PSNR / MS-SSIM: 22.0217 / 0.7367
    LPIPS / DISTS: 0.5551 / 0.3474
```

This supports trying a simple decoder-side gate predicted from semantic latent,
detail context, and base-condition statistics. The gate must be fully
deterministic from decoded payload/model state, or else any transmitted gate
bits must be counted in `actual_payload_bpp`.

Deterministic RGB output-gate diagnostic:

```text
implementation:
  src/coserdic/models/gencodec_backbone.py
  scripts/train_stage4_cod_lite_gate.py
  scripts/eval_stage4_cod_lite_adapter.py --gate-checkpoint

adapter:
  20260628_stage4_cod_lite_pyramid_sem256_detailctx6_statsmatch_ft2k_b8.pt

gate checkpoint:
  checkpoints/stage4_cod_lite_gate/20260628_stage4_cod_lite_gate_statsmatch_fidelity_600_b1ga2.pt

train:
  20260628_stage4_cod_lite_gate_statsmatch_fidelity_600_b1ga2

eval:
  20260628_stage4_cod_lite_gate_statsmatch_fidelity_600_b1ga2_full552_eval

patch-FID saved-image audit:
  20260628_stage4_cod_lite_gate_statsmatch_fidelity_600_b1ga2_full552_patchfid256
```

Full552 metrics at unchanged actual_payload_bpp:

```text
actual_payload_bpp:
  0.013999

Stage 3:
  PSNR / MS-SSIM: 21.9951 / 0.7348
  LPIPS / DISTS: 0.5758 / 0.3536
  patch-FID256:   146.5134

learned deterministic gate:
  alpha mean/std/min/max: 0.3071 / 0.0773 / 0.1494 / 0.4414
  PSNR / MS-SSIM:        22.0367 / 0.7384
  LPIPS / DISTS:         0.5369 / 0.3481
  patch-FID256:          109.3629
```

Comparison to the fixed stats-match alpha0.30 anchor:

```text
fixed alpha0.30:
  PSNR / MS-SSIM: 22.0421 / 0.7377
  LPIPS / DISTS:  0.5372 / 0.3489
  patch-FID256:   110.5424

learned gate:
  PSNR is slightly lower
  MS-SSIM, LPIPS, DISTS, and patch-FID are slightly better
```

Diagnostic interpretation:

```text
the learned RGB output gate is not a Stage 5 result and does not close the
external CoD-Lite gap

it is a useful no-extra-bit diagnostic for the fixed-alpha tradeoff, but it is
still post-hoc RGB output blending and should not be promoted as the method
```

Mainline condition-space gate:

```text
policy:
  025_stage4_no_posthoc_rgb_mainline_condition_gate.md

implementation:
  CoSERToCoDLiteConditionGate
  scripts/train_stage4_cod_lite_condition_gate.py
  scripts/eval_stage4_cod_lite_adapter.py --condition-gate-checkpoint

train:
  20260628_stage4_cod_lite_condition_gate_fidelity_probe300_b1ga2

eval:
  20260628_stage4_cod_lite_condition_gate_fidelity_probe300_b1ga2_full552_eval
```

Full552 metrics at unchanged actual_payload_bpp, with no RGB output blend:

```text
actual_payload_bpp:
  0.013999

Stage 3:
  PSNR / MS-SSIM: 21.9951 / 0.7348
  LPIPS / DISTS: 0.5758 / 0.3536

condition gate:
  condition_gate_mean: 0.4327
  PSNR / MS-SSIM:     21.3169 / 0.7161
  LPIPS / DISTS:      0.5255 / 0.3437
```

Mainline interpretation:

```text
condition-space gate is the correct mechanism class, but this first probe is
not faithful enough and is not promoted

the result confirms that RGB blending was hiding a real condition-control gap
inside the CoD-Lite integration
```

Implementation caution:

```text
the official CoD-Lite image decode path currently fails for batch_size > 1 in
this wrapper, so gate image-loss training/evaluation should use batch_size=1
with gradient accumulation until the wrapper is made batch-safe
```

## Condition-Stat Diagnosis

Full552 condition-stat run:

```text
run:
  20260628_stage4_condition_stats_full552

checkpoint:
  20260628_stage4_cod_lite_pyramid_sem256_detailctx6_tanh075_traincache2048_5k_b8.pt

count:
  552

actual_payload_bpp:
  0.013999

base_to_target_l1:
  0.5371

pred_to_target_l1:
  0.4112

pred L1 win rate over base:
  0.9982

base_to_target_relative_l2:
  0.8758

pred_to_target_relative_l2:
  0.6716

target / base / pred condition std:
  0.7954 / 0.8234 / 0.6932

target / base / pred spatial high-frequency ratio:
  0.2575 / 0.2776 / 0.2280
```

The semantic-latent + detail-context adapter is genuinely moving CoD-Lite
condition tensors toward the reference-native condition. However, the predicted
condition is lower-energy and lower-frequency than the target condition. The
next adapter training should therefore add condition cosine/stat/spectrum
matching in addition to L1, and promotion should still be judged by final
decoded images.

Condition-stat matching follow-up:

```text
train:
  20260628_stage4_cod_lite_pyramid_sem256_detailctx6_statsmatch_ft2k_b8

init:
  20260628_stage4_cod_lite_pyramid_sem256_detailctx6_tanh075_traincache2048_5k_b8.pt

checkpoint:
  checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_statsmatch_ft2k_b8.pt

loss additions:
  condition cosine: 0.25
  condition channel stats: 0.20
  condition high-frequency ratio: 0.05
```

Condition-stat effect on full552:

```text
pred_to_target_l1:
  0.4112 -> 0.4165

pred_to_target_cosine:
  0.7447 -> 0.7463

pred condition std:
  0.6932 -> 0.7282
  target: 0.7954

pred spatial high-frequency ratio:
  0.2280 -> 0.2349
  target: 0.2575
```

Full552 image metrics with patch-FID256:

```text
alpha  PSNR     MS-SSIM  LPIPS    DISTS    patch-FID
0.00   21.9951  0.7348   0.5758   0.3536   146.5134
0.25   22.0549  0.7378   0.5467   0.3510   117.3462
0.30   22.0421  0.7377   0.5372   0.3489   110.5424
0.40   21.9920  0.7368   0.5185   0.3439    98.2211
0.50   21.9103  0.7350   0.5011   0.3381    88.2460
0.75   21.5835  0.7264   0.4655   0.3214    72.1553
1.00   21.1181  0.7125   0.4428   0.3039    63.7198
```

Delta versus the previous detail-context checkpoint:

```text
alpha  dPSNR    dMS-SSIM  dLPIPS   dDISTS   dFID
0.25   +0.0001  +0.0014   -0.0041  -0.0024  -1.95
0.30   -0.0010  +0.0017   -0.0049  -0.0030  -2.22
0.40   -0.0044  +0.0022   -0.0059  -0.0040  -2.50
1.00   -0.0496  +0.0062   -0.0060  -0.0089  -1.45
```

Per-split alpha0.30 after stats matching:

```text
Stage 3 / alpha0.30:
  Kodak24:
    PSNR 21.6674 -> 21.6741
    MS-SSIM 0.7225 -> 0.7250
    LPIPS 0.6366 -> 0.5925
    DISTS 0.3732 -> 0.3713

  CLIC2020 test428:
    PSNR 22.3638 -> 22.4204
    MS-SSIM 0.7454 -> 0.7486
    LPIPS 0.5610 -> 0.5231
    DISTS 0.3499 -> 0.3448

  DIV2K val100:
    PSNR 20.4956 -> 20.5112
    MS-SSIM 0.6927 -> 0.6944
    LPIPS 0.6245 -> 0.5845
    DISTS 0.3648 -> 0.3610
```

Interpretation:

```text
condition-stat matching slightly worsens condition L1 but improves condition
energy/frequency alignment and full552 decoded-image metrics. The alpha0.25
and alpha0.30 stability anchors are now cleaner per split, but they remain
diagnostics. The method should move next to a deterministic learned gate and a
larger train cache, not more fixed-alpha hand tuning.
```

## Baseline Gap

Official CoD-Lite Kodak24 512 baseline at comparable rate:

```text
0.0156 checkpoint:
  actual_payload_bpp: 0.015625
  PSNR / MS-SSIM: 20.7667 / 0.7090
  LPIPS / DISTS / FID: 0.2259 / 0.1402 / 38.4785
```

Current conclusion:

```text
CoSER-DiC Stage 4 semantic-latent + detail-context adapter improves the internal
Stage 3 codec and gives a stronger decoder-side Pareto curve, but it does not
yet beat official CoD-Lite at comparable bpp.
The research bottleneck remains high-quality CoD/CoD-Lite condition recovery
and/or a tiny counted control stream in Stage 5.
```

## Next Mainline Steps

1. Increase Stage-4 adapter training scale beyond the 2048-image train cache,
   preferably adding DIV2K train and more OpenImages while keeping the 552-image
   CoD reproduction split untouched for evaluation.
2. Improve the adapter or gating so the chosen alpha improves Kodak, CLIC2020
   test, and DIV2K val consistently, not only the aggregate full552 table.
3. Compare the detail-context adapter against official CoD-Lite checkpoints per
   rate and per dataset using the same actual-payload bpp policy and patch-FID
   settings.
4. Try simple Stage 5 counted control streams only after the decoder-side
   semantic-latent adapter saturates:
   tiny condition residual tokens, rate-control map, or semantic confidence map.
5. Promote only if gains are measured against official CoD-Lite/CoD checkpoints
   using clearly labeled `actual_payload_bpp`.
