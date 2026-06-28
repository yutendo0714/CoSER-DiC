# Stage 4 Decoder-Side Refiner

Date: 2026-06-28 JST  
Status: Archived diagnostic; not part of the active MVP Stage 4 path  
Parent: `docs/research/design_decisions/019_stage1_rateprior_b16_reconnection.md`
Superseded by:
`docs/research/design_decisions/021_stage4_cod_codlite_parallel_backbone_policy.md`

## Current Status

The ResUNet-style decoder refiner is no longer used to steer the Core MVP
Stage 4 architecture. Its results are retained as engineering diagnostics only:

```text
useful:
  actual-bpp-preserving decoder-side evaluation scaffold
  conditioning I/O smoke test
  warning that LPIPS/DISTS gains can regress full-dataset FID

not sufficient:
  does not validate CoD or CoD-Lite diffusion decoder behavior
  does not replace the MVP requirement for a CoD-Lite/CoD diffusion backbone
```

Active Stage 4 work should use the CoD-Lite default track and CoD parallel
heavy track described in Decision 021.

All sections below are retained only for reproducibility and audit. Do not use
the old refiner commands as the next Stage 4 plan unless the archived branch is
explicitly revived by a new decision memo.

## Decision

Add a fixed decoder-side refiner as the first Stage 4 model-side improvement
path.

The refiner preserves the Core MVP bitstream axis:

```text
input:
  decoded Stage 3 image
  decoded semantic-only image
  decoded low-resolution detail residual
  decoded semantic latent reconstructed from entropy-decoded semantic tokens

output:
  refined image

payload policy:
  fixed decoder-side weights are not image-specific side information.
  actual_payload_bpp / paper_bpp is unchanged from the Stage 3 payload.
  if future per-image controls, masks, prompts, maps, or seeds are transmitted,
  they must be counted in actual_payload_bpp.
```

This is deliberately not a wholesale import of an external diffusion codec. It
is a CoSER-owned module that can later be upgraded toward a stronger
generative/perceptual decoder while keeping the semantic/detail entropy split.

## Implementation

```text
model:
  src/coserdic/models/decoder_refiner.py
  DecoderSideRefiner
  DecoderSideRefinerConfig

training:
  scripts/train_stage4_decoder_refiner.py
  configs/train/train_stage4_decoder_refiner.yaml

evaluation integration:
  scripts/eval_stage3_uniform_residual_bitstream.py
  optional --decoder-refiner-checkpoint

visual audit:
  scripts/export_worst_case_gallery.py
  now supports stage4_image links

tests:
  tests/test_decoder_refiner.py
  tests/test_export_worst_case_gallery.py
```

The output convolution is zero-initialized, so a fresh refiner starts as an
identity mapping. This avoids accidentally degrading the current Stage 3 anchor
before training.

The default training configuration now uses a guarded probe schedule:

```text
residual_scale: 0.03
batch_size: 4
max_steps: 1000
checkpoint_every: 250
l1: 0.35
ms_ssim: 0.25
lpips: 0.20
anchor_stage3_l1: 0.50
refiner_tv: 0.10
refiner_chroma: 0.20
train_roots:
  /dpl/open-images-v6/train/data
```

This was tightened after the 1k step500 candidate improved LPIPS/DISTS but
regressed full CLIC2020/DIV2K patch FID. Longer Stage 4 training should now be
launched only with an explicit command after a guarded candidate passes the
full validation gate.

The OpenImages-only default is intentional. The local `/dpl/div2k` directory is
a flat mixed 0001-0900 directory; using it directly in training risks leaking
DIV2K validation images into the model-side refiner while DIV2K val is part of
the CoD-style evaluation protocol.

## Batch Size Probe

The initial Stage 4 runs used `batch_size=2`. On the local RTX 4070 Ti SUPER,
the mid-guard configuration was probed at larger 512x512 batches:

```text
batch_size=4:
  run:
    20260628_stage4_decoder_refiner_bs4_probe_anchor025_rs005_tv005_chroma010_oi_only
  wandb:
    wandb/offline-run-20260628_125533-a0ekumeh
  result:
    passed 10 training steps and wrote a checkpoint.

batch_size=8:
  run:
    20260628_stage4_decoder_refiner_bs8_probe_anchor025_rs005_tv005_chroma010_oi_only
  wandb:
    wandb/offline-run-20260628_125557-pi2b6hkx
  result:
    failed with CUDA OOM during the refiner forward pass.

decision:
  Use batch_size=4 for the next long Stage 4 refiner run on this machine.
  Keep batch_size=2 as the fallback if extra metrics or a larger refiner are
  enabled later.
```

## Smoke Verification

Training smoke:

```text
run:
  20260628_stage4_decoder_refiner_smoke2

wandb:
  wandb/offline-run-20260628_114448-khjxkhny

checkpoint:
  checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_smoke2.pt

settings:
  crop_size: 128
  batch_size: 1
  max_steps: 2
  loss: L1 + Stage3 anchor only for smoke

result:
  checkpoint, sample image, summary JSON, experiment doc, and wandb offline run
  were created successfully.
```

Actual-bitstream evaluation smoke:

```text
run:
  20260628_stage4_decoder_refiner_smoke2_cod512_bitstream_eval1

wandb:
  wandb/offline-run-20260628_114927-a4sne4xb

summary:
  results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_smoke2_cod512_bitstream_eval1/summary.json

protocol:
  cod_reproduction_512
  crop_size: 512
  max_images: 1

roundtrip:
  semantic: true
  detail: true
  failure_count: 0

actual_payload_bpp / paper_bpp:
  0.014587 on the single smoke image

Stage 3:
  PSNR / MS-SSIM: 20.141619 / 0.585184

Stage 4 smoke refiner:
  PSNR / MS-SSIM: 20.141165 / 0.585180
  delta vs Stage 3: -0.000454 / -0.000004

interpretation:
  This 2-step checkpoint is not a quality result. It only validates that the
  refiner can be evaluated after entropy decode without changing payload bpp.
```

Checkpoint-every smoke:

```text
run:
  20260628_stage4_decoder_refiner_checkpoint_every_smoke2
wandb:
  wandb/offline-run-20260628_132020-tpn3u5vt
settings:
  crop_size: 128
  batch_size: 1
  max_steps: 2
  checkpoint_every: 1
result:
  wrote the final checkpoint and the intermediate checkpoint:
    checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_checkpoint_every_smoke2_step000001.pt
  CUDA remained visible and the run finished successfully.
```

## Training Stability Note

The first 200-step LPIPS/MS-SSIM run failed immediately with a non-finite AMP
gradient. The refiner forward path can remain under autocast, but the loss
composition is now forced to float32:

```text
failed run:
  20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200

symptom:
  FloatingPointError: non-finite gradient norm at step 1: nan

fix:
  compute L1, MS-SSIM, LPIPS, DISTS, and Stage3-anchor losses outside AMP.

verified stable:
  20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms10_fp32loss
  20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_fp32loss
  20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_anchor020_rs005
  20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms300_anchor030_rs004_tv020_chroma030_oi_only
  20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms300_anchor025_rs005_tv005_chroma010_oi_only
```

## Short-Run Results

All results below use actual entropy-coded semantic/detail payload bpp from the
Stage 3 bitstream. The fixed refiner weights are decoder-side model parameters,
not image-specific side information, so `actual_payload_bpp / paper_bpp` is
unchanged.

Base operating point:

```text
semantic prior:
  topk2048 learned escape Huffman, prefix-replay decoder-safe

detail prior:
  static semantic-position-left-context residual Huffman
  d32, b4, range0.25

decoder:
  detail_gain: 0.5
  unsharp3x3 strength: 0.15
```

Aggressive 200-step run:

```text
run:
  20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_fp32loss
wandb:
  wandb/offline-run-20260628_115541-q5bt8p4f
checkpoint:
  checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_fp32loss.pt
settings:
  residual_scale: 0.125
  anchor_stage3_l1: 0.05

Kodak24 actual-bitstream eval:
  run:
    20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_kodak24_eval
  actual_payload_bpp: 0.014120738
  roundtrip: semantic=true, detail=true, failure_count=0
  Stage3 PSNR / MS-SSIM / LPIPS / DISTS:
    21.562603 / 0.721580 / 0.633858 / 0.375367
  Stage4 PSNR / MS-SSIM / LPIPS / DISTS:
    21.446821 / 0.716920 / 0.555626 / 0.348684
  delta vs Stage3:
    PSNR -0.115782, MS-SSIM -0.004660,
    LPIPS -0.078232, DISTS -0.026684
  per-image wins:
    LPIPS 24/24, DISTS 24/24, PSNR 0/24, MS-SSIM 0/24
  Kodak64 torchmetrics patch-FID:
    Stage4 210.063766
```

Conservative 200-step run:

```text
run:
  20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_anchor020_rs005
wandb:
  wandb/offline-run-20260628_120059-nmra09pl
checkpoint:
  checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_anchor020_rs005.pt
settings:
  residual_scale: 0.05
  anchor_stage3_l1: 0.20
training caveat:
  this run used the pre-update training root list, which included flat
  /dpl/div2k. Keep as a diagnostic, not as the preferred Stage 4 evidence.

Kodak24 actual-bitstream eval:
  run:
    20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_anchor020_rs005_kodak24_eval
  actual_payload_bpp: 0.014120738
  roundtrip: semantic=true, detail=true, failure_count=0
  Stage3 PSNR / MS-SSIM / LPIPS / DISTS:
    21.562603 / 0.721580 / 0.633858 / 0.375367
  Stage4 PSNR / MS-SSIM / LPIPS / DISTS:
    21.498006 / 0.717667 / 0.562431 / 0.349110
  delta vs Stage3:
    PSNR -0.064597, MS-SSIM -0.003913,
    LPIPS -0.071427, DISTS -0.026257
  per-image wins:
    LPIPS 24/24, DISTS 24/24, PSNR 0/24, MS-SSIM 0/24
  Kodak64 torchmetrics patch-FID:
    Stage3 gain0.50 baseline: 215.036667
    Stage4 conservative:      197.914581
```

CLIC64 conservative spot-check:

```text
run:
  20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_anchor020_rs005_clic64_eval
actual_payload_bpp: 0.013593197
roundtrip: semantic=true, detail=true, failure_count=0
Stage3 PSNR / MS-SSIM / LPIPS / DISTS:
  22.381245 / 0.772970 / 0.498252 / 0.336193
Stage4 PSNR / MS-SSIM / LPIPS / DISTS:
  22.296666 / 0.769695 / 0.464864 / 0.318584
delta vs Stage3:
  PSNR -0.084579, MS-SSIM -0.003275,
  LPIPS -0.033388, DISTS -0.017609
per-image wins:
  LPIPS 55/64, DISTS 59/64, PSNR 0/64, MS-SSIM 1/64
CLIC64 torchmetrics patch-FID, same subset:
  Stage3: 174.378281
  Stage4: 176.662582
```

Leakage-safe OpenImages-only 200-step rerun, preferred short-run evidence:

```text
run:
  20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_anchor020_rs005_oi_only
wandb:
  wandb/offline-run-20260628_121703-b34ut8kf
checkpoint:
  checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_anchor020_rs005_oi_only.pt
settings:
  train_roots: /dpl/open-images-v6/train/data
  residual_scale: 0.05
  anchor_stage3_l1: 0.20

Kodak24 actual-bitstream eval:
  run:
    20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_anchor020_rs005_oi_only_kodak24_eval
  wandb:
    wandb/offline-run-20260628_121948-yfqsn6mp
  actual_payload_bpp: 0.014120738
  roundtrip: semantic=true, detail=true, failure_count=0
  Stage3 PSNR / MS-SSIM / LPIPS / DISTS:
    21.562603 / 0.721580 / 0.633858 / 0.375367
  Stage4 PSNR / MS-SSIM / LPIPS / DISTS:
    21.506064 / 0.718158 / 0.580210 / 0.358177
  delta vs Stage3:
    PSNR -0.056539, MS-SSIM -0.003422,
    LPIPS -0.053648, DISTS -0.017190
  per-image wins:
    LPIPS 24/24, DISTS 24/24, PSNR 0/24, MS-SSIM 0/24
  Kodak64 torchmetrics patch-FID, same subset:
    Stage3: 215.036469
    Stage4: 209.338760

CLIC64 actual-bitstream eval:
  run:
    20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_anchor020_rs005_oi_only_clic64_eval
  wandb:
    wandb/offline-run-20260628_122407-ey0w3dd5
  actual_payload_bpp: 0.013593197
  roundtrip: semantic=true, detail=true, failure_count=0
  Stage3 PSNR / MS-SSIM / LPIPS / DISTS:
    22.381245 / 0.772970 / 0.498252 / 0.336193
  Stage4 PSNR / MS-SSIM / LPIPS / DISTS:
    22.302456 / 0.770030 / 0.470095 / 0.322705
  delta vs Stage3:
    PSNR -0.078789, MS-SSIM -0.002940,
    LPIPS -0.028158, DISTS -0.013488
  per-image wins:
    LPIPS 59/64, DISTS 59/64, PSNR 0/64, MS-SSIM 0/64
  CLIC64 torchmetrics patch-FID, same subset:
    Stage3: 174.378281
    Stage4: 174.435394
```

Guarded OpenImages-only 300-step sweep:

```text
strong guard run:
  20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms300_anchor030_rs004_tv020_chroma030_oi_only
settings:
  residual_scale: 0.04
  anchor_stage3_l1: 0.30
  refiner_tv: 0.20
  refiner_chroma: 0.30

Kodak24:
  actual_payload_bpp: 0.014120738
  Stage4 delta vs Stage3:
    PSNR -0.007971, MS-SSIM -0.000368,
    LPIPS -0.014120, DISTS -0.004677
  Kodak64 torchmetrics patch-FID:
    Stage3: 215.036469
    Stage4: 215.787201

CLIC64:
  actual_payload_bpp: 0.013593197
  Stage4 delta vs Stage3:
    PSNR -0.014887, MS-SSIM -0.000357,
    LPIPS -0.009350, DISTS -0.003573
  CLIC64 torchmetrics patch-FID:
    Stage3: 174.378281
    Stage4: 174.310165

interpretation:
  This is fidelity-safe but too weak as a perceptual refiner. It validates that
  TV/chroma guards can suppress refiner side effects, but it underuses the
  Stage 4 capacity and does not improve Kodak FID.

mid guard run:
  20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms300_anchor025_rs005_tv005_chroma010_oi_only
settings:
  residual_scale: 0.05
  anchor_stage3_l1: 0.25
  refiner_tv: 0.05
  refiner_chroma: 0.10

Kodak24 actual-bitstream eval:
  run:
    20260628_stage4_decoder_refiner_midguard_rs005_tv005_chroma010_oi_only_kodak24_eval
  wandb:
    wandb/offline-run-20260628_124728-j3utgzra
  actual_payload_bpp: 0.014120738
  roundtrip: semantic=true, detail=true, failure_count=0
  Stage4 PSNR / MS-SSIM / LPIPS / DISTS:
    21.527073 / 0.720120 / 0.571894 / 0.359198
  delta vs Stage3:
    PSNR -0.035530, MS-SSIM -0.001460,
    LPIPS -0.061964, DISTS -0.016170
  per-image wins:
    LPIPS 24/24, DISTS 23/24, PSNR 0/24, MS-SSIM 0/24
  Kodak64 torchmetrics patch-FID:
    Stage3: 215.036469
    Stage4: 199.045944

CLIC64 actual-bitstream eval:
  run:
    20260628_stage4_decoder_refiner_midguard_rs005_tv005_chroma010_oi_only_clic64_eval
  wandb:
    wandb/offline-run-20260628_125147-hwz5ctb1
  actual_payload_bpp: 0.013593197
  roundtrip: semantic=true, detail=true, failure_count=0
  Stage4 PSNR / MS-SSIM / LPIPS / DISTS:
    22.328170 / 0.771557 / 0.461705 / 0.323259
  delta vs Stage3:
    PSNR -0.053074, MS-SSIM -0.001414,
    LPIPS -0.036547, DISTS -0.012934
  per-image wins:
    LPIPS 61/64, DISTS 58/64, PSNR 0/64, MS-SSIM 0/64
  CLIC64 torchmetrics patch-FID:
    Stage3: 174.378281
    Stage4: 174.321960

interpretation:
  This is the best balanced short-run Stage 4 candidate so far. It preserves the
  unchanged actual_payload_bpp property, improves LPIPS/DISTS strongly, improves
  Kodak patch-FID substantially, and no longer shows the CLIC64 FID regression
  seen in the earlier conservative run. The remaining cost is a consistent
  PSNR/MS-SSIM drop, so this point should be presented as rate-perception
  oriented, not as an RD-only improvement.
```

Mid-guard 2k-step batch-4 run:

```text
run:
  20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms2k_bs4_anchor025_rs005_tv005_chroma010_oi_only
wandb:
  wandb/offline-run-20260628_125728-mog1qbd4
checkpoint:
  checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms2k_bs4_anchor025_rs005_tv005_chroma010_oi_only.pt
settings:
  batch_size: 4
  max_steps: 2000
  residual_scale: 0.05
  anchor_stage3_l1: 0.25
  refiner_tv: 0.05
  refiner_chroma: 0.10

Kodak24 actual-bitstream eval:
  run:
    20260628_stage4_decoder_refiner_ms2k_bs4_midguard_oi_only_kodak24_eval
  wandb:
    wandb/offline-run-20260628_131030-r0fmva8f
  actual_payload_bpp: 0.014120738
  roundtrip: semantic=true, detail=true, failure_count=0
  Stage4 PSNR / MS-SSIM / LPIPS / DISTS:
    21.444545 / 0.713628 / 0.484011 / 0.333425
  delta vs Stage3:
    PSNR -0.118058, MS-SSIM -0.007952,
    LPIPS -0.149847, DISTS -0.041942
  per-image wins:
    LPIPS 24/24, DISTS 24/24, PSNR 0/24, MS-SSIM 0/24
  Kodak64 torchmetrics patch-FID:
    Stage3: 215.036469
    Stage4: 165.095108

CLIC64 actual-bitstream eval:
  run:
    20260628_stage4_decoder_refiner_ms2k_bs4_midguard_oi_only_clic64_eval
  wandb:
    wandb/offline-run-20260628_131449-jpkwhd44
  actual_payload_bpp: 0.013593197
  roundtrip: semantic=true, detail=true, failure_count=0
  Stage4 PSNR / MS-SSIM / LPIPS / DISTS:
    22.230607 / 0.765213 / 0.413388 / 0.310564
  delta vs Stage3:
    PSNR -0.150638, MS-SSIM -0.007757,
    LPIPS -0.084864, DISTS -0.025630
  per-image wins:
    LPIPS 57/64, DISTS 55/64, PSNR 0/64, MS-SSIM 1/64
  CLIC64 torchmetrics patch-FID:
    Stage3: 174.378281
    Stage4: 186.570114

interpretation:
  The longer run is a stronger perception-biased operating point, not the new
  default. It substantially improves LPIPS/DISTS and Kodak patch-FID, but it
  over-pushes the refiner on CLIC64: PSNR/MS-SSIM drop much more than the
  300-step mid-guard point and CLIC64 patch-FID regresses. This suggests that
  long Stage 4 training needs checkpoint selection and CLIC/DIV2K validation
  gates rather than blindly taking the final checkpoint.
```

Checkpoint-selection 1k sweep:

```text
run:
  20260628_stage4_decoder_refiner_ckpt250_1k_bs4_anchor025_rs005_tv005_chroma010_oi_only
wandb:
  wandb/offline-run-20260628_132136-92e0hwy5
checkpoints:
  step250:
    checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_ckpt250_1k_bs4_anchor025_rs005_tv005_chroma010_oi_only_step000250.pt
  step500:
    checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_ckpt250_1k_bs4_anchor025_rs005_tv005_chroma010_oi_only_step000500.pt
  step750:
    checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_ckpt250_1k_bs4_anchor025_rs005_tv005_chroma010_oi_only_step000750.pt
  step1000:
    checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_ckpt250_1k_bs4_anchor025_rs005_tv005_chroma010_oi_only.pt

CLIC64 checkpoint screen:
  Stage3 patch-FID baseline:
    174.378281
  step250:
    delta PSNR / MS-SSIM / LPIPS / DISTS:
      -0.073260 / -0.002857 / -0.051800 / -0.018209
    wins:
      LPIPS 60/64, DISTS 58/64, PSNR 0/64, MS-SSIM 1/64
    CLIC128 torchmetrics patch-FID:
      180.647430
  step500:
    delta PSNR / MS-SSIM / LPIPS / DISTS:
      -0.071926 / -0.002724 / -0.056711 / -0.017118
    wins:
      LPIPS 62/64, DISTS 56/64, PSNR 0/64, MS-SSIM 1/64
    CLIC128 torchmetrics patch-FID:
      172.581406
  step750:
    delta PSNR / MS-SSIM / LPIPS / DISTS:
      -0.090251 / -0.004066 / -0.067543 / -0.020312
    wins:
      LPIPS 60/64, DISTS 56/64, PSNR 0/64, MS-SSIM 1/64
    CLIC128 torchmetrics patch-FID:
      175.158325
  step1000:
    delta PSNR / MS-SSIM / LPIPS / DISTS:
      -0.115118 / -0.004917 / -0.075625 / -0.025473
    wins:
      LPIPS 59/64, DISTS 57/64, PSNR 0/64, MS-SSIM 1/64
    CLIC128 torchmetrics patch-FID:
      178.079422

Kodak24 confirmation for step500:
  run:
    20260628_stage4_decoder_refiner_ckpt250sweep_step500_kodak24_eval
  wandb:
    wandb/offline-run-20260628_134749-hspq36a8
  actual_payload_bpp:
    0.014120738
  roundtrip:
    semantic=true, detail=true, failure_count=0
  delta PSNR / MS-SSIM / LPIPS / DISTS:
    -0.049467 / -0.002532 / -0.092378 / -0.021814
  wins:
    LPIPS 24/24, DISTS 23/24, PSNR 0/24, MS-SSIM 1/24
  Kodak64 torchmetrics patch-FID:
    Stage3: 215.036469
    Stage4: 177.938614

decision:
  step500 is the best current Stage 4 checkpoint-selection candidate. It
  improves CLIC64 patch-FID against Stage 3, improves Kodak24 patch-FID more
  strongly than the 300-step candidate, and keeps the fidelity drop below the
  2k perception-heavy checkpoint. It still needs full Kodak24 + CLIC2020 test
  428 + DIV2K val100 evaluation before promotion.
```

Full 552-image strict check for step500:

```text
run:
  20260628_stage4_decoder_refiner_ckpt250sweep_step500_cod512_full552_eval

checkpoint:
  checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_ckpt250_1k_bs4_anchor025_rs005_tv005_chroma010_oi_only_step000500.pt

protocol:
  cod_reproduction_512
  Kodak24 + CLIC2020 test 428 + DIV2K val100
  512x512 resize/center-crop
  actual entropy-coded payload bpp

roundtrip:
  semantic=true, detail=true, failure_count=0

overall:
  actual_payload_bpp: 0.013998612
  delta PSNR / MS-SSIM / LPIPS / DISTS:
    -0.063862 / -0.002519 / -0.074130 / -0.018497

per-dataset image metrics:
  Kodak24:
    bpp 0.014120738
    delta PSNR / MS-SSIM / LPIPS / DISTS:
      -0.049467 / -0.002532 / -0.092378 / -0.021814
    wins:
      LPIPS 24/24, DISTS 23/24
  CLIC2020 test 428:
    bpp 0.013746172
    delta PSNR / MS-SSIM / LPIPS / DISTS:
      -0.065516 / -0.002552 / -0.071146 / -0.018188
    wins:
      LPIPS 423/428, DISTS 385/428
  DIV2K val100:
    bpp 0.015049744
    delta PSNR / MS-SSIM / LPIPS / DISTS:
      -0.060235 / -0.002378 / -0.082522 / -0.019025
    wins:
      LPIPS 100/100, DISTS 99/100

GenCodec public-script patch FID:
  backend:
    torchmetrics.image.fid.FrechetInceptionDistance
  patch protocol:
    Kodak64, non-Kodak256, split=2 half-shift
  Kodak24:
    Stage3 215.036469 -> Stage4 177.938614
    delta -37.097855
  CLIC2020 test 428:
    Stage3 147.601395 -> Stage4 170.253677
    delta +22.652283
  DIV2K val100:
    Stage3 235.171097 -> Stage4 262.204926
    delta +27.033829

decision:
  Do not promote this checkpoint. It is a useful perception-improving
  diagnostic and it wins LPIPS/DISTS broadly at unchanged actual_payload_bpp,
  but it fails the full-dataset FID gate on CLIC2020 test and DIV2K val.
```

Interpretation:

```text
positive signal:
  The refiner strongly improves LPIPS/DISTS at unchanged actual_payload_bpp.
  The balanced 300-step mid-guard OpenImages-only run improves Kodak64
  patch-FID from 215.036469 to 199.045944 and slightly improves CLIC64
  patch-FID from 174.378281 to 174.321960 on the same 64-image subset.
  Checkpoint selection improves this tradeoff: the 1k sweep step500 checkpoint
  reaches Kodak64 patch-FID 177.938614 and improves CLIC64 under the exploratory
  subset screen while preserving exact roundtrip and unchanged actual_payload_bpp.
  The 2k-step run shows that the same module has much larger perceptual
  capacity: Kodak64 patch-FID reaches 165.095108 and LPIPS/DISTS improve
  strongly, without changing actual_payload_bpp.

negative signal:
  PSNR and MS-SSIM consistently drop.
  Full 552-image FID is now the central caution. Step500 improves Kodak64 FID
  but regresses CLIC2020 test and DIV2K val under the public GenCodec-style
  Kodak64/non-Kodak256 patch FID. The subset CLIC64 screen was therefore not a
  sufficient promotion gate.
  Visual audit shows sharper structure but a risk of halo/color-fringe-like
  outlines around high-contrast edges and textured regions.

current decision:
  Keep the OpenImages-only ckpt250 sweep step500 checkpoint as a diagnostic
  candidate only:
    checkpoint:
      checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_ckpt250_1k_bs4_anchor025_rs005_tv005_chroma010_oi_only_step000500.pt
    residual_scale=0.05
    anchor_stage3_l1=0.25
    refiner_tv=0.05
    refiner_chroma=0.10
  Keep the 300-step checkpoint as historical balanced evidence, and treat the
  2k-step checkpoint as a perception-heavy diagnostic candidate.
  Do not promote Stage 4 yet. The next Stage 4 attempt must preserve the
  broad LPIPS/DISTS gains while preventing CLIC2020/DIV2K FID regression,
  likely via a more conservative residual scale/loss schedule and a full
  validation gate rather than a 64-image subset gate.
```

## Next Guarded Candidate

Do not launch the previous 20k-step Stage 4 schedule yet. The step500 full
strict check showed that subset CLIC64 screening can miss CLIC2020/DIV2K FID
regression. The next run should be a more conservative probe that limits the
decoder-side residual while keeping the Core MVP bitstream unchanged.

Recommended short-to-mid guarded probe:

```bash
.venv/bin/python scripts/train_stage4_decoder_refiner.py \
  --config configs/train/train_stage4_decoder_refiner.yaml \
  --run-name 20260628_stage4_decoder_refiner_guarded_rs003_lpips020_ms025_anchor050_tv010_chroma020_oi_only_1k \
  --crop-size 512 \
  --batch-size 4 \
  --max-steps 1000 \
  --checkpoint-every 250 \
  --residual-scale 0.03 \
  --loss-l1 0.35 \
  --loss-ms-ssim 0.25 \
  --loss-lpips 0.20 \
  --loss-dists 0.00 \
  --loss-anchor-stage3 0.50 \
  --loss-refiner-tv 0.10 \
  --loss-refiner-chroma 0.20 \
  --sample-every 250 \
  --log-every 50 \
  --wandb-mode offline
```

Screen intermediate checkpoints cheaply on Kodak24 + CLIC64, but use this only
as an early rejection gate. Promotion requires the full 552-image strict
evaluation and per-dataset GenCodec public-script patch FID:

```bash
.venv/bin/python scripts/eval_stage3_uniform_residual_bitstream.py \
  --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt \
  --semantic-prior outputs/stage2_learned_entropy/20260628_stage2_learned_topk2048_escape_huffman_fit_32768tokens_crop512_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json \
  --token-prior-checkpoint checkpoints/stage2_token_prior/20260628_stage2_token_prior_crop512_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_amp_es.pt \
  --detail-codec semantic_position_leftctx_huffman \
  --detail-prior outputs/stage3_residual_entropy/20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_crop512_8192calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json \
  --eval-protocol cod_reproduction_512 \
  --crop-size 512 \
  --batch-size 1 \
  --detail-downsample-factor 32 \
  --detail-bits 4 \
  --detail-range 0.25 \
  --detail-gain 0.5 \
  --decoder-postprocess unsharp3x3 \
  --decoder-postprocess-strength 0.15 \
  --decoder-refiner-checkpoint checkpoints/stage4_decoder_refiner/<selected_stage4_checkpoint>.pt \
  --compute-perceptual \
  --save-reconstructions \
  --save-reconstruction-limit 552 \
  --run-name 20260628_stage4_decoder_refiner_guarded_rs003_selected_full552_eval \
  --wandb-mode offline
```

Then compute torchmetrics FID on the Stage 3 and Stage 4 reconstruction
directories, per dataset, with the public GenCodec-style patch protocol:
Kodak64/non-Kodak256 and split=2 half-shift.

Checkpoint selection rule:

```text
save:
  every 500 steps plus the final checkpoint

screen:
  Kodak24 + CLIC64 subset with actual_payload_bpp and patch-FID only for early
  rejection

promote only if:
  roundtrip_failure_count == 0
  actual_payload_bpp is unchanged from Stage 3
  LPIPS/DISTS improve on Kodak24, CLIC2020 test, and DIV2K val
  CLIC2020 test and DIV2K val patch-FID do not regress against Stage 3
  PSNR/MS-SSIM drop is no worse than the intended rate-perception operating
  point

if final checkpoint regresses but an earlier checkpoint passes:
  select the earlier checkpoint and document it; do not take the final step by
  default.
```

Guarded probe result:

```text
run:
  20260628_stage4_decoder_refiner_guarded_rs003_lpips020_ms025_anchor050_tv010_chroma020_lr5e5_oi_only_1k

wandb:
  wandb/offline-run-20260628_143719-1qfbtow4

checkpoints:
  step250, step500, step750, step1000/final

training:
  residual_scale: 0.03
  lr: 5e-5
  l1 / ms_ssim / lpips / dists:
    0.35 / 0.25 / 0.20 / 0.00
  anchor_stage3_l1 / tv / chroma:
    0.50 / 0.10 / 0.20

CLIC64 actual-bitstream screen:
  Stage3 patch256 FID baseline:
    227.199387
  step250:
    bpp 0.013593197
    delta PSNR / MS-SSIM / LPIPS / DISTS:
      -0.016141 / -0.000618 / -0.009830 / -0.006407
    CLIC patch256 FID:
      228.728561, delta +1.529175
  step500:
    bpp 0.013593197
    delta PSNR / MS-SSIM / LPIPS / DISTS:
      -0.037136 / -0.001737 / -0.044482 / -0.017230
    CLIC patch256 FID:
      240.693832, delta +13.494446
  step750:
    bpp 0.013593197
    delta PSNR / MS-SSIM / LPIPS / DISTS:
      -0.044784 / -0.002366 / -0.054488 / -0.017597
    CLIC patch256 FID:
      241.990082, delta +14.790695
  step1000:
    bpp 0.013593197
    delta PSNR / MS-SSIM / LPIPS / DISTS:
      -0.052518 / -0.002545 / -0.058262 / -0.017781
    CLIC patch256 FID:
      244.556274, delta +17.356888

decision:
  Do not promote the guarded refiner either. Conservative residual scaling and
  stronger anchoring reduce distortion loss, but CLIC patch256 FID still
  regresses. The next Stage 4 attempt should not be another simple loss-weight
  sweep of the same refiner. It needs a design change such as frequency-aware
  residual constraints, decoder-side identity gating, or moving the perceptual
  improvement into a better Stage 3 detail representation/entropy point.
```

## Research Risk

```text
main upside:
  attacks the observed Stage 3 weakness: structured high-frequency content.

main risk:
  a small refiner may optimize LPIPS locally while worsening FID or introducing
  texture artifacts. Do not promote it without strict 512 actual-bpp evaluation,
  patch FID, per-image failure analysis, and visual galleries.

stage boundary:
  Stage 4 is not complete until it beats the current Stage 3 operating points
  under actual_payload_bpp with exact roundtrip.
```
