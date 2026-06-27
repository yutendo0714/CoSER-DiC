# Stage 3 Residual Progress

Date: 2026-06-27 JST

## Stage1 LPIPS Ft500 32k Entropy Reconnection

The 4k ft500 reconnection looked promising perceptually but paid too much
semantic bpp. I completed the 32k token export, low-LR transformer prior, top-k
escape Huffman fit, and ft500-specific b4 residual recalibration.

Artifacts:

```text
token export:
  outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_32768_oi_div2k_from_stage1_lpips002_ft500_fullroots/
  wandb: wandb/offline-run-20260627_185231-k4yrji4g

token prior:
  checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_stage1_lpips002_ft500_lr1e4_do02_es.pt
  wandb: wandb/offline-run-20260627_185255-bty0et9l

semantic top-k prior:
  outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_stage1_lpips002_ft500_lowlr8k/
  wandb: wandb/offline-run-20260627_190948-opdn8c13

residual prior:
  outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_stage1_lpips002_ft500/
  wandb: wandb/offline-run-20260627_191628-tjvnaajt
```

Key entropy numbers:

```text
ft500 32768-token global_entropy_bits: 12.281880
ft500 32768-token prior val bits:      10.343893
topk512 hit / escape:                   0.652379 / 0.347621
calibration mean payload bpp:           0.010480
```

Final CoSER common LIC candidate in this block:

```text
LPIPS-first run:
  20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_lpips002_ft500_32kprior_8192resid_pp_unsharp200_perceptual
wandb:
  wandb/offline-run-20260627_191835-4a8p89bo

actual_payload_bpp: 0.016248
semantic_bpp:       0.011025
detail_bpp:         0.005223
PSNR / MS-SSIM:    21.217353 / 0.694143
LPIPS / DISTS:      0.597244 / 0.374431
roundtrip:          semantic=true, detail=true
```

Safer variants at the same actual payload bpp:

```text
none:
  run: 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_lpips002_ft500_32kprior_8192resid_perceptual
  wandb: wandb/offline-run-20260627_192703-v8pgpwvu
  PSNR / MS-SSIM: 21.298414 / 0.695166
  LPIPS / DISTS: 0.626042 / 0.387196

unsharp1:
  run: 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_lpips002_ft500_32kprior_8192resid_pp_unsharp100_perceptual
  wandb: wandb/offline-run-20260627_192510-gmx7wmf5
  PSNR / MS-SSIM: 21.262786 / 0.694748
  LPIPS / DISTS: 0.611016 / 0.379559
```

Comparison against active b5/unsharp2:

```text
active b5/unsharp2 actual_payload_bpp: 0.018370
candidate bpp delta:                  -0.002122
candidate LPIPS delta:                -0.082515
candidate DISTS delta:                -0.024799
candidate PSNR delta:                 -0.127305

per-image wins:
  lower bpp:     161 / 165
  lower LPIPS:   163 / 165
  lower DISTS:   156 / 165
  higher PSNR:    38 / 165
```

Distribution metrics:

```text
image FID/KID:     289.087749 / 0.107389
patch128 FID/KID: 235.915591 / 0.122006
```

Interpretation:

```text
This is now a serious rate-perception candidate: it is lower-rate than active
b5/unsharp2 and much better on LPIPS/DISTS. It is not an RD replacement because
PSNR drops and FID/KID remain worse than the active unsharp2 branch. The next
research pressure should be entropy-aware Stage1 tuning or a stronger token
prior, plus safer residual/postprocess settings for distribution quality.
```

## Stage1 LPIPS Ft500 d384/l6 Semantic Prior Update

The d256/l4 32k prior recovered much of the ft500 entropy penalty, but the
validation NLL still lagged the active branch. I trained a stronger d384/l6
token prior and re-fit the same top-k512 escape Huffman interface.

Artifacts:

```text
token prior:
  checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf384_l6_12kstep_32768tokens_stage1_lpips002_ft500_lr1e4_do02_es.pt
  wandb: wandb/offline-run-20260627_192905-99zdasv2

semantic top-k prior:
  outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_stage1_lpips002_ft500_tf384_l6_12k/
  wandb: wandb/offline-run-20260627_194728-0e3l9l4y
```

Entropy comparison:

```text
d256/l4 topk512:
  topk_hit / escape:          0.652379 / 0.347621
  bits_per_token:             10.676670
  calibration payload bpp:    0.010480

d384/l6 topk512:
  topk_hit / escape:          0.718921 / 0.281079
  bits_per_token:             10.106184
  calibration payload bpp:    0.009923

d384/l6 topk256:
  topk_hit / escape:          0.607009 / 0.392991
  bits_per_token:             10.360242
  calibration payload bpp:    0.010171
  decision:                   reject; worse than topk512

d384/l6 topk1024:
  topk_hit / escape:          0.828103 / 0.171897
  bits_per_token:              9.948924
  calibration payload bpp:    0.009769
  decision:                   strong; run Stage3

d384/l6 topk2048:
  topk_hit / escape:          0.921675 / 0.078325
  bits_per_token:              9.825860
  calibration payload bpp:    0.009649
  wandb:                      wandb/offline-run-20260627_214048-j2ofqg13
  decision:                   best payload bpp, but much slower to fit/measure
```

CoSER common LIC, same Stage1 checkpoint, b4 residual prior, and fixed
postprocess settings:

```text
shared:
  actual_payload_bpp: 0.015985
  semantic_bpp:       0.010761
  detail_bpp:         0.005223
  semantic_topk_hit:  0.732102
  roundtrip:          semantic=true, detail=true

none:
  run: 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_lpips002_ft500_tf384_topk1024_32kprior_8192resid_perceptual
  wandb: wandb/offline-run-20260627_203802-o0mbxyay
  PSNR / MS-SSIM: 21.298414 / 0.695166
  LPIPS / DISTS: 0.626042 / 0.387196

unsharp1:
  run: 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_lpips002_ft500_tf384_topk1024_32kprior_8192resid_pp_unsharp100_perceptual
  wandb: wandb/offline-run-20260627_203526-vhgg7k36
  PSNR / MS-SSIM: 21.262786 / 0.694748
  LPIPS / DISTS: 0.611016 / 0.379559

unsharp2:
  run: 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_lpips002_ft500_tf384_topk1024_32kprior_8192resid_pp_unsharp200_perceptual
  wandb: wandb/offline-run-20260627_203243-scti6si0
  PSNR / MS-SSIM: 21.217353 / 0.694143
  LPIPS / DISTS: 0.597244 / 0.374431

topk2048 none:
  run: 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_lpips002_ft500_tf384_topk2048_32kprior_8192resid_perceptual
  wandb: wandb/offline-run-20260627_215931-4osw5ifz
  actual_payload_bpp: 0.015850
  semantic_bpp:       0.010627
  detail_bpp:         0.005223
  semantic_topk_hit:  0.855398
  PSNR / MS-SSIM:     21.298414 / 0.695166
  LPIPS / DISTS:      0.626042 / 0.387196
  roundtrip:          semantic=true, detail=true

topk2048 unsharp1:
  run: 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_lpips002_ft500_tf384_topk2048_32kprior_8192resid_pp_unsharp100_perceptual
  wandb: wandb/offline-run-20260627_220229-7003wutq
  actual_payload_bpp: 0.015850
  semantic_bpp:       0.010627
  detail_bpp:         0.005223
  semantic_topk_hit:  0.855398
  PSNR / MS-SSIM:     21.262786 / 0.694748
  LPIPS / DISTS:      0.611016 / 0.379559
  roundtrip:          semantic=true, detail=true

topk2048 unsharp2:
  run: 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_lpips002_ft500_tf384_topk2048_32kprior_8192resid_pp_unsharp200_perceptual
  wandb: wandb/offline-run-20260627_214350-wk0otkqf
  actual_payload_bpp: 0.015850
  semantic_bpp:       0.010627
  detail_bpp:         0.005223
  semantic_topk_hit:  0.855398
  PSNR / MS-SSIM:     21.217353 / 0.694143
  LPIPS / DISTS:      0.597244 / 0.374431
  roundtrip:          semantic=true, detail=true
```

Interpretation:

```text
This is a clean entropy-side improvement. Reconstruction metrics are unchanged
because the Stage1 checkpoint, residual quantizer, and postprocess are fixed,
but actual_payload_bpp drops from 0.016248 to 0.015985 with topk1024 and to
0.015850 with topk2048 versus the d256 32k prior. Topk256 was too small,
topk512 was good, topk1024 was efficient, and topk2048 gave the best actual
Stage3 payload while being much slower to fit/measure. Against active
b5/unsharp2, the ft500 b4/unsharp2 candidate is now 0.002520 bpp lower while
preserving the same LPIPS/DISTS advantage. The next autonomous pressure should
shift from top-k width to residual/FID tuning or entropy-aware Stage1
regularization.
```

GenCodec reproduction protocol for the same topk sweep:

```text
strict protocol:
  Kodak 24
  CLIC2020 test 428 = professional/test 250 + mobile/test 178
  DIV2K val 100 = 0801.png-0900.png

topk1024 unsharp2:
  run: 20260627_stage3_b4_semposleft_g4_gencodec_reproduction_stage1_lpips002_ft500_tf384_topk1024_32kprior_8192resid_pp_unsharp200_perceptual
  wandb: wandb/offline-run-20260627_205054-fjbibejz
  actual_payload_bpp: 0.015409
  semantic_bpp:       0.010236
  detail_bpp:         0.005173
  PSNR / MS-SSIM:     21.931414 / 0.723483
  LPIPS / DISTS:      0.562911 / 0.364492

topk2048 unsharp2:
  run: 20260627_stage3_b4_semposleft_g4_gencodec_reproduction_stage1_lpips002_ft500_tf384_topk2048_32kprior_8192resid_pp_unsharp200_perceptual
  wandb: wandb/offline-run-20260627_215239-g5mfih6n
  actual_payload_bpp: 0.015293
  semantic_bpp:       0.010120
  detail_bpp:         0.005173
  PSNR / MS-SSIM:     21.931414 / 0.723483
  LPIPS / DISTS:      0.562911 / 0.364492
  roundtrip:          semantic=true, detail=true
```

## Stage1 LPIPS Ft500 Topk2048 Detail Gain Probe

After locking the stronger topk2048 semantic payload, I checked whether a fixed
decoder `detail_gain` can reduce the FID/KID regression without changing the
actual transmitted payload. This keeps the Core MVP structure intact: semantic
tokens and residual codes are unchanged, while the decoder applies a fixed
gain to the decoded residual grid.

CoSER common LIC, same checkpoint/prior/residual stream:

```text
shared payload:
  actual_payload_bpp: 0.015850
  semantic_bpp:       0.010627
  detail_bpp:         0.005223
  semantic_topk_hit:  0.855398
  roundtrip:          semantic=true, detail=true

gain0.5 none:
  run: 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_lpips002_ft500_tf384_topk2048_32kprior_8192resid_gain050_perceptual
  recon export: 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_lpips002_ft500_tf384_topk2048_32kprior_8192resid_gain050_recon_export
  wandb: wandb/offline-run-20260627_220803-2o77kk4f
  PSNR / MS-SSIM:     21.179101 / 0.694154
  LPIPS / DISTS:      0.623141 / 0.389397
  image FID / KID:    282.674784 / 0.090455
  patch128 FID / KID: 227.783365 / 0.104939

gain0.75 none:
  run: 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_lpips002_ft500_tf384_topk2048_32kprior_8192resid_gain075_perceptual
  recon export: 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_lpips002_ft500_tf384_topk2048_32kprior_8192resid_gain075_recon_export
  wandb: wandb/offline-run-20260627_221055-sa6tkorz
  PSNR / MS-SSIM:     21.273588 / 0.694981
  LPIPS / DISTS:      0.623817 / 0.388134
  image FID / KID:    282.912345 / 0.091588
  patch128 FID / KID: 228.972532 / 0.107392

gain1.0 none baseline:
  PSNR / MS-SSIM:     21.298414 / 0.695166
  LPIPS / DISTS:      0.626042 / 0.387196
  image FID / KID:    284.082780 / 0.096637
  patch128 FID / KID: 230.860580 / 0.112192
```

Interpretation:

```text
detail_gain=0.5 is a real FID/KID-safe fixed decoder point. It improves both
image and patch distribution metrics versus gain1.0 none and nearly reaches
the active b5 image FID, but it sacrifices most of the LPIPS/DISTS benefit.
detail_gain=0.75 is in between but is worse than gain0.5 on all measured
distribution metrics. Recommended reporting split is now:

  gain0.5 none:
    FID/KID-safe point
  gain1.0 unsharp1:
    balanced perceptual point
  gain1.0 unsharp2:
    LPIPS/DISTS-first point
```

## Evaluation Protocol Correction

Added named evaluation protocol resolution after auditing `/dpl`.

```text
CLIC2020 test:
  /dpl/clic/professional/test = 250
  /dpl/clic/mobile/test       = 178
  total                       = 428

DIV2K:
  /dpl/div2k is a flat mixed 0001.png-0900.png directory.
  Evaluation must use 0801.png-0900.png for DIV2K validation.
```

New resolver and report command:

```text
src/coserdic/datasets/eval_protocols.py
scripts/dataset_protocol_report.py
configs/eval/protocols.yaml
```

## Protocol Perceptual Evaluation

Added optional LPIPS/DISTS evaluation for Stage 1/2/3 bitstream scripts and
CompressAI anchors:

```text
src/coserdic/metrics/perceptual.py
```

CoSER common LIC protocol, 165 images:

```text
b5 quality anchor:
  run: 20260627_stage3_b5_semposleft_g4_coser_common_lic_perceptual
  actual_payload_bpp: 0.018368
  PSNR delta: +0.471572 dB
  LPIPS delta: -0.005412
  DISTS delta: -0.005049

b4 low-rate anchor:
  run: 20260627_stage3_b4_semposleft_g4_coser_common_lic_perceptual
  actual_payload_bpp: 0.015678
  PSNR delta: +0.395093 dB
  LPIPS delta: -0.000832
  DISTS delta: -0.005221
```

GenCodec reproduction protocol, 552 images:

```text
b5 quality anchor:
  run: 20260627_stage3_b5_semposleft_g4_gencodec_reproduction_perceptual
  actual_payload_bpp: 0.017736
  PSNR delta: +0.554524 dB
  LPIPS delta: -0.006018
  DISTS delta: -0.004718
```

Decision memo:

```text
docs/research/design_decisions/013_stage3_protocol_perceptual_evaluation.md
```

## Fixed Decoder Postprocess Probe

Added a deterministic fixed decoder postprocess path for Stage 3:

```text
src/coserdic/models/postprocess.py
scripts/eval_stage3_uniform_residual_bitstream.py
```

The postprocess is applied after semantic reconstruction plus decoded residual
grid. It is a fixed decoder configuration and does not change
`actual_payload_bpp`; per-image postprocess selection would need a transmitted
control stream and must be counted.

Best current perceptual preset:

```text
decoder_postprocess: unsharp3x3
decoder_postprocess_strength: 2.0
run: 20260627_stage3_b5_semposleft_g4_coser_common_lic_pp_unsharp200_perceptual
wandb: wandb/offline-run-20260627_173309-1st4cvof
recon export: 20260627_stage3_b5_semposleft_g4_coser_common_lic_pp_unsharp200_recon_export
recon export wandb: wandb/offline-run-20260627_173614-7nfcvxk4
```

CoSER common LIC, 165 images:

```text
neutral b5:
  actual_payload_bpp: 0.018370
  PSNR / MS-SSIM: 21.390054 / 0.693741
  LPIPS / DISTS: 0.694714 / 0.409676

unsharp3x3 strength 2.0:
  actual_payload_bpp: 0.018370
  PSNR / MS-SSIM: 21.344658 / 0.693968
  LPIPS / DISTS: 0.679759 / 0.399230
  LPIPS wins vs neutral: 163 / 165
  DISTS wins vs neutral: 163 / 165
```

Metric-max diagnostic:

```text
decoder_postprocess: unsharp3x3
decoder_postprocess_strength: 4.0
run: 20260627_stage3_b5_semposleft_g4_coser_common_lic_pp_unsharp400_perceptual
wandb: wandb/offline-run-20260627_174457-o1t7uyfo
recon export: 20260627_stage3_b5_semposleft_g4_coser_common_lic_pp_unsharp400_recon_export
recon export wandb: wandb/offline-run-20260627_174734-k0kmpz7l

actual_payload_bpp: 0.018370
PSNR / MS-SSIM: 21.274349 / 0.693789
LPIPS / DISTS: 0.663952 / 0.391251
LPIPS wins vs neutral: 163 / 165
DISTS wins vs neutral: 158 / 165
```

Distribution metrics for unsharp3x3 strength 2.0:

```text
image FID/KID: 282.247165 / 0.082436
patch128 FID/KID: 222.928785 / 0.107602
```

Distribution metrics for unsharp3x3 strength 4.0:

```text
image FID/KID: 279.442998 / 0.082808
patch128 FID/KID: 229.295739 / 0.122919
```

Interpretation:

```text
Keep decoder_postprocess=none as the neutral anchor.
Keep unsharp3x3 strength=2.0 as a safer fixed perceptual preset.
Treat unsharp3x3 strength=4.0 as a metric-max diagnostic for now because
patch-level distribution metrics worsen versus 2.0.
Do not claim this as learned Stage 5 progress; it is a useful no-payload
diagnostic that motivates a learned perceptual refinement decoder.
```

Decision memo:

```text
docs/research/design_decisions/017_stage3_fixed_decoder_unsharp_postprocess.md
```

## Starting Point

Stage 2 active semantic stream:

```text
learned top512/escape Huffman
prior:
  outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json
```

Stage 3 had config scaffolding but no actual detail-bitstream implementation.

## Implemented

Added a fixed actual-byte residual bootstrap and a decoder-known static
Huffman residual payload:

```text
src/coserdic/entropy/residual_grid.py
scripts/fit_stage3_residual_huffman_prior.py
scripts/eval_stage3_uniform_residual_bitstream.py
tests/test_residual_grid.py
```

The codec sends a low-resolution uniformly quantized residual grid through
`CoSERBitstream.detail_latents`. The decoder reconstructs:

```text
semantic image + bilinear-upsampled residual grid
```

This preserves the Core MVP stream separation:

```text
semantic tokens: learned top-k/escape bitstream
detail latents: residual-grid bitstream
```

## Key Result

Best low-bpp bootstrap setting from the quick sweep before residual entropy
coding:

```text
detail_downsample_factor: 32
detail_bits: 4
detail_range: 0.25
detail_codec: zlib_fixed_bits
```

Kodak:

```text
run: 20260627_stage3_uniform_residual_d32_b4_r025_zlib_kodak_eval
stats rerun: 20260627_stage3_uniform_residual_d32_b4_r025_zlib_kodak_eval_stats
wandb: wandb/offline-run-20260627_113714-g5te9p9x

semantic_payload_bpp: 0.010722
detail_payload_bpp: 0.008759
total_payload_bpp: 0.019480
stage3_full_stream_bpp: 0.080907

semantic-only PSNR / MS-SSIM: 20.7363 / 0.67577
stage3 PSNR / MS-SSIM: 21.1068 / 0.68042
delta: +0.3705 dB / +0.00465 MS-SSIM

semantic token roundtrip: true
detail code roundtrip: true
roundtrip_failure_count: 0

residual_grid_abs_mean: 0.02022
residual_grid_std: 0.02511
residual_grid_clip_ratio: 0.0
detail_code_entropy_bits: 1.6878
```

The same d32/b4/r0.25 residual symbols were then coded with a decoder-known
static Huffman prior fit on 4096 OpenImages + DIV2K calibration crops:

```text
fit run: 20260627_stage3_residual_huffman_d32_b4_r025_4096calib_oi_div2k
wandb: wandb/offline-run-20260627_114352-rrbpxgyp
prior:
  outputs/stage3_residual_entropy/20260627_stage3_residual_huffman_d32_b4_r025_4096calib_oi_div2k/static_residual_grid_huffman_prior.json

num_images: 4096
symbol_entropy_bits: 1.9339
mean_huffman_bits_per_symbol: 2.0087
fixed_bits_per_symbol: 4
clip_ratio: 0.000136
```

Kodak static-Huffman actual-byte result:

```text
run: 20260627_stage3_residual_huffman_d32_b4_r025_4096calib_kodak_eval
wandb: wandb/offline-run-20260627_114414-j6e6y0la

semantic_payload_bpp: 0.010722
detail_payload_bpp: 0.005544
total_payload_bpp: 0.016266
stage3_full_stream_bpp: 0.077693

semantic-only PSNR / MS-SSIM: 20.7363 / 0.67577
stage3 PSNR / MS-SSIM: 21.1068 / 0.68042
delta: +0.3705 dB / +0.00465 MS-SSIM

semantic token roundtrip: true
detail code roundtrip: true
roundtrip_failure_count: 0
```

Cross-dataset checks:

```text
DIV2K first 100:
  run: 20260627_stage3_uniform_residual_d32_b4_r025_zlib_div2k100_eval
  wandb: wandb/offline-run-20260627_113413-s5ernjrm
  total_payload_bpp: 0.019026
  PSNR delta: +0.3055 dB
  MS-SSIM delta: +0.00403
  roundtrip_failure_count: 0

CLIC professional valid:
  run: 20260627_stage3_uniform_residual_d32_b4_r025_zlib_clicval64_eval
  wandb: wandb/offline-run-20260627_113437-rybb6lqc
  total_payload_bpp: 0.017995
  PSNR delta: +0.5649 dB
  MS-SSIM delta: +0.00246
  roundtrip_failure_count: 0
```

Static-Huffman cross-dataset checks:

```text
DIV2K first 100:
  run: 20260627_stage3_residual_huffman_d32_b4_r025_4096calib_div2k100_eval
  wandb: wandb/offline-run-20260627_114454-qwwjhqye
  semantic_payload_bpp: 0.010425
  detail_payload_bpp: 0.005948
  total_payload_bpp: 0.016373
  full_stream_bpp: 0.077795
  PSNR delta: +0.3055 dB
  MS-SSIM delta: +0.00403
  roundtrip_failure_count: 0

CLIC professional valid:
  run: 20260627_stage3_residual_huffman_d32_b4_r025_4096calib_clicval64_eval
  wandb: wandb/offline-run-20260627_114527-wglryta2
  semantic_payload_bpp: 0.009906
  detail_payload_bpp: 0.005443
  total_payload_bpp: 0.015348
  full_stream_bpp: 0.076758
  PSNR delta: +0.5649 dB
  MS-SSIM delta: +0.00246
  roundtrip_failure_count: 0
```

## Static-Huffman RD Probe

After the d32/b4/r0.25 result, the same actual-bitstream path was used to probe
nearby residual quantizers. This avoids overfitting the Stage 3 decision to the
zlib-era sweep.

Kodak:

```text
d32 b4 r0.50:
  fit run: 20260627_stage3_residual_huffman_d32_b4_r050_4096calib_oi_div2k
  eval run: 20260627_stage3_residual_huffman_d32_b4_r050_4096calib_kodak_eval
  wandb eval: wandb/offline-run-20260627_115347-ebo6u4pi
  detail_payload_bpp: 0.004425
  total_payload_bpp: 0.015147
  full_stream_bpp: 0.076574
  PSNR delta: +0.2309 dB
  MS-SSIM delta: -0.00037
  roundtrip_failure_count: 0

d32 b5 r0.25:
  fit run: 20260627_stage3_residual_huffman_d32_b5_r025_4096calib_oi_div2k
  eval run: 20260627_stage3_residual_huffman_d32_b5_r025_4096calib_kodak_eval
  wandb eval: wandb/offline-run-20260627_115138-kbyixk0n
  detail_payload_bpp: 0.008286
  total_payload_bpp: 0.019007
  full_stream_bpp: 0.080434
  PSNR delta: +0.3957 dB
  MS-SSIM delta: +0.00576
  roundtrip_failure_count: 0

d32 b6 r0.25:
  fit run: 20260627_stage3_residual_huffman_d32_b6_r025_4096calib_oi_div2k
  eval run: 20260627_stage3_residual_huffman_d32_b6_r025_4096calib_kodak_eval
  wandb eval: wandb/offline-run-20260627_115243-2h5dhyss
  detail_payload_bpp: 0.011256
  total_payload_bpp: 0.021978
  full_stream_bpp: 0.083420
  PSNR delta: +0.4015 dB
  MS-SSIM delta: +0.00605
  roundtrip_failure_count: 0
```

The b5/r0.25 point was cross-checked because it is the best quality/rate
candidate under 0.02 total payload bpp:

```text
DIV2K first 100:
  run: 20260627_stage3_residual_huffman_d32_b5_r025_4096calib_div2k100_eval
  wandb: wandb/offline-run-20260627_115432-4fwrcjoh
  semantic_payload_bpp: 0.010425
  detail_payload_bpp: 0.008558
  total_payload_bpp: 0.018983
  full_stream_bpp: 0.080410
  PSNR delta: +0.4601 dB
  MS-SSIM delta: +0.00537
  roundtrip_failure_count: 0

CLIC professional valid:
  run: 20260627_stage3_residual_huffman_d32_b5_r025_4096calib_clicval64_eval
  wandb: wandb/offline-run-20260627_115503-558577cj
  semantic_payload_bpp: 0.009906
  detail_payload_bpp: 0.007970
  total_payload_bpp: 0.017876
  full_stream_bpp: 0.079286
  PSNR delta: +0.7171 dB
  MS-SSIM delta: +0.00370
  roundtrip_failure_count: 0
```

## Position-Conditioned Residual Huffman

The residual-grid distribution diagnostics showed that per-channel/position
entropy is lower than global entropy. A decoder-known channel/position Huffman
codec was added for the same quantized residual grid, with no reconstruction
change and no per-image side information:

```text
implementation: StaticResidualGridPositionHuffmanCode
fit mode: scripts/fit_stage3_residual_huffman_prior.py --coding-mode position_huffman
eval mode: scripts/eval_stage3_uniform_residual_bitstream.py --detail-codec position_huffman
```

Kodak:

```text
d32 b5 r0.25 position Huffman:
  fit run: 20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_oi_div2k
  eval run: 20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_kodak_eval
  wandb eval: wandb/offline-run-20260627_122510-kqs89ydi
  detail_payload_bpp: 0.008250
  total_payload_bpp: 0.018972
  full_stream_bpp: 0.080399
  PSNR delta: +0.3957 dB
  MS-SSIM delta: +0.00576
  roundtrip_failure_count: 0

d32 b4 r0.25 position Huffman:
  fit run: 20260627_stage3_residual_poshuff_d32_b4_r025_4096calib_oi_div2k
  eval run: 20260627_stage3_residual_poshuff_d32_b4_r025_4096calib_kodak_eval
  wandb eval: wandb/offline-run-20260627_122621-3s1aqboe
  detail_payload_bpp: 0.005483
  total_payload_bpp: 0.016205
  full_stream_bpp: 0.077632
  PSNR delta: +0.3705 dB
  MS-SSIM delta: +0.00465
  roundtrip_failure_count: 0
```

Cross-dataset checks:

```text
b5 position Huffman:
  DIV2K first 100:
    run: 20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_div2k100_eval
    wandb: wandb/offline-run-20260627_122905-dgjk98rq
    detail_payload_bpp: 0.008549
    total_payload_bpp: 0.018973
    PSNR delta: +0.4601 dB
    MS-SSIM delta: +0.00537
  CLIC professional valid:
    run: 20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_clicval64_eval
    wandb: wandb/offline-run-20260627_122939-uuzfdcy1
    detail_payload_bpp: 0.007967
    total_payload_bpp: 0.017873
    PSNR delta: +0.7171 dB
    MS-SSIM delta: +0.00370

b4 position Huffman:
  DIV2K first 100:
    run: 20260627_stage3_residual_poshuff_d32_b4_r025_4096calib_div2k100_eval
    wandb: wandb/offline-run-20260627_122711-j8ni48q7
    detail_payload_bpp: 0.005837
    total_payload_bpp: 0.016262
    PSNR delta: +0.3055 dB
    MS-SSIM delta: +0.00403
  CLIC professional valid:
    run: 20260627_stage3_residual_poshuff_d32_b4_r025_4096calib_clicval64_eval
    wandb: wandb/offline-run-20260627_122803-0m0wsksb
    detail_payload_bpp: 0.005267
    total_payload_bpp: 0.015172
    PSNR delta: +0.5649 dB
    MS-SSIM delta: +0.00246
```

Intermediate decision at this point: promote position Huffman to the Stage 3
residual entropy path.
The gain is small, especially for b5, but it is consistent across Kodak,
DIV2K, and CLIC while preserving the same decoded residual grid. At this stage
the quality bootstrap was d32/b5/r0.25 position Huffman and the low-rate anchor
was d32/b4/r0.25 position Huffman. Later hybrid results supersede both pure
position anchors. The older global Huffman results remain useful as a simpler
ablation.

## Compact CoSERBitstream Header

The JSON outer stream header was a large fixed cost for 256x256 crop
evaluation. A compact binary header was added behind
`CoSERBitstream(header_codec="compact")`; the existing JSON stream remains
decodable and is still available as `header_codec="json"`.

Kodak compact-header actual full stream bpp:

```text
d32 b5 r0.25 position Huffman:
  run: 20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_kodak_eval_compacthdr
  wandb: wandb/offline-run-20260627_123512-dq2t1e7v
  stream_header_codec: compact
  semantic_payload_bpp: 0.010722
  detail_payload_bpp: 0.008250
  total_payload_bpp: 0.018972
  semantic_only_full_stream_bpp: 0.027812
  stage3_full_stream_bpp: 0.037282
  previous JSON-header stage3_full_stream_bpp: 0.080399
  roundtrip_failure_count: 0

d32 b4 r0.25 position Huffman:
  run: 20260627_stage3_residual_poshuff_d32_b4_r025_4096calib_kodak_eval_compacthdr
  wandb: wandb/offline-run-20260627_123537-95rgqsve
  stream_header_codec: compact
  semantic_payload_bpp: 0.010722
  detail_payload_bpp: 0.005483
  total_payload_bpp: 0.016205
  semantic_only_full_stream_bpp: 0.027812
  stage3_full_stream_bpp: 0.034515
  previous JSON-header stage3_full_stream_bpp: 0.077632
  roundtrip_failure_count: 0
```

Compact-header cross-dataset checks:

```text
d32 b5 r0.25 position Huffman:
  DIV2K first 100:
    run: 20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_div2k100_eval_compacthdr
    wandb: wandb/offline-run-20260627_124113-wigh652w
    detail_payload_bpp: 0.008549
    total_payload_bpp: 0.018973
    stage3_full_stream_bpp: 0.037284
    PSNR delta: +0.4601 dB
    MS-SSIM delta: +0.00537
    roundtrip_failure_count: 0
  CLIC professional valid:
    run: 20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_clicval64_eval_compacthdr
    wandb: wandb/offline-run-20260627_124142-lsatlbwp
    num_images: 41
    detail_payload_bpp: 0.007967
    total_payload_bpp: 0.017873
    stage3_full_stream_bpp: 0.036183
    PSNR delta: +0.7171 dB
    MS-SSIM delta: +0.00370
    roundtrip_failure_count: 0

d32 b4 r0.25 position Huffman:
  DIV2K first 100:
    run: 20260627_stage3_residual_poshuff_d32_b4_r025_4096calib_div2k100_eval_compacthdr
    wandb: wandb/offline-run-20260627_124219-ugd3uztm
    detail_payload_bpp: 0.005837
    total_payload_bpp: 0.016262
    stage3_full_stream_bpp: 0.034573
    PSNR delta: +0.3055 dB
    MS-SSIM delta: +0.00403
    roundtrip_failure_count: 0
  CLIC professional valid:
    run: 20260627_stage3_residual_poshuff_d32_b4_r025_4096calib_clicval64_eval_compacthdr
    wandb: wandb/offline-run-20260627_124247-eryfeq3k
    num_images: 41
    detail_payload_bpp: 0.005267
    total_payload_bpp: 0.015172
    stage3_full_stream_bpp: 0.033483
    PSNR delta: +0.5649 dB
    MS-SSIM delta: +0.00246
    roundtrip_failure_count: 0
```

This is a container improvement, not a model-quality improvement. It matters
because the evaluation requirement is actual transmitted bytes; the compact
header makes full-stream bpp a more meaningful number for low-bitrate crop
experiments while preserving checksums and length-delimited payloads.

## CRC32 Compact Checksum

The compact stream still carried a 32-byte SHA256 checksum, which is a visible
fixed cost in 256x256 low-bpp evaluation. `CoSERBitstream` now supports an
explicit `checksum_codec="crc32"` mode. SHA256 remains the default and old
compact streams remain decodable.

Implementation and tests:

```text
src/coserdic/entropy/bitstream.py
tests/test_bitstream.py

new eval flags:
  --stream-header-codec compact
  --stream-checksum-codec crc32

test status:
  45 passed, 1 warning
```

Stage 2 semantic-only Kodak check:

```text
run: 20260627_stage2_learned_topk512_escape_huffman_32768tokens_8kprior_kodak_eval_decodersched_compacthdr_crc32_shortid
wandb: wandb/offline-run-20260627_130140-djajfext
learned payload_bpp: 0.010722
learned compact_crc32 short-id full_stream_bpp: 0.021830
fixed_bits payload_bpp: 0.012695
fixed_bits compact_crc32 short-id full_stream_bpp: 0.023682
roundtrip_failure_count: 0
```

Stage 3 compact CRC32 actual full stream:

```text
active d32 b5 r0.25 position Huffman:
  Kodak:
    run: 20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_kodak_eval_compacthdr_crc32
    wandb: wandb/offline-run-20260627_125032-8zoimefl
    total_payload_bpp: 0.018972
    stage3_full_stream_bpp: 0.033864
    PSNR delta: +0.3957 dB
    MS-SSIM delta: +0.00576
  DIV2K first 100:
    run: 20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_div2k100_eval_compacthdr_crc32
    wandb: wandb/offline-run-20260627_125212-d6o802xx
    total_payload_bpp: 0.018973
    stage3_full_stream_bpp: 0.033866
    PSNR delta: +0.4601 dB
    MS-SSIM delta: +0.00537
  CLIC professional valid:
    run: 20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_clicval64_eval_compacthdr_crc32
    wandb: wandb/offline-run-20260627_125238-0u6i8ugo
    total_payload_bpp: 0.017873
    stage3_full_stream_bpp: 0.032765
    PSNR delta: +0.7171 dB
    MS-SSIM delta: +0.00370

low-rate d32 b4 r0.25 position Huffman:
  Kodak:
    run: 20260627_stage3_residual_poshuff_d32_b4_r025_4096calib_kodak_eval_compacthdr_crc32
    wandb: wandb/offline-run-20260627_125101-cpql516l
    total_payload_bpp: 0.016205
    stage3_full_stream_bpp: 0.031097
    PSNR delta: +0.3705 dB
    MS-SSIM delta: +0.00465
  DIV2K first 100:
    run: 20260627_stage3_residual_poshuff_d32_b4_r025_4096calib_div2k100_eval_compacthdr_crc32
    wandb: wandb/offline-run-20260627_125317-h68rvzaf
    total_payload_bpp: 0.016262
    stage3_full_stream_bpp: 0.031155
    PSNR delta: +0.3055 dB
    MS-SSIM delta: +0.00403
  CLIC professional valid:
    run: 20260627_stage3_residual_poshuff_d32_b4_r025_4096calib_clicval64_eval_compacthdr_crc32
    wandb: wandb/offline-run-20260627_125345-oo08oouo
    total_payload_bpp: 0.015172
    stage3_full_stream_bpp: 0.030065
    PSNR delta: +0.5649 dB
    MS-SSIM delta: +0.00246

roundtrip_failure_count: 0 for all Stage 3 compact CRC32 runs
```

The evaluation scripts now transmit short codec IDs instead of long experiment
names in the compact header:

```text
s2lte0: Stage 2 learned top-k escape
s2sth0: Stage 2 static entropy ablation
s3urg0: Stage 3 uniform residual grid
s3rae0: Stage 3 residual AE
```

Stage 3 compact CRC32 + short-ID actual full stream:

```text
active d32 b5 r0.25 position Huffman:
  Kodak:
    run: 20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_kodak_eval_compacthdr_crc32_shortid
    wandb: wandb/offline-run-20260627_125830-wke3i8sf
    total_payload_bpp: 0.018972
    stage3_full_stream_bpp: 0.031301
  DIV2K first 100:
    run: 20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_div2k100_eval_compacthdr_crc32_shortid
    wandb: wandb/offline-run-20260627_125941-0qgk0vg6
    total_payload_bpp: 0.018973
    stage3_full_stream_bpp: 0.031302
  CLIC professional valid:
    run: 20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_clicval64_eval_compacthdr_crc32_shortid
    wandb: wandb/offline-run-20260627_130007-om6jmrfq
    total_payload_bpp: 0.017873
    stage3_full_stream_bpp: 0.030202

low-rate d32 b4 r0.25 position Huffman:
  Kodak:
    run: 20260627_stage3_residual_poshuff_d32_b4_r025_4096calib_kodak_eval_compacthdr_crc32_shortid
    wandb: wandb/offline-run-20260627_125856-19zqaznl
    total_payload_bpp: 0.016205
    stage3_full_stream_bpp: 0.028534
  DIV2K first 100:
    run: 20260627_stage3_residual_poshuff_d32_b4_r025_4096calib_div2k100_eval_compacthdr_crc32_shortid
    wandb: wandb/offline-run-20260627_130047-owfc51s0
    total_payload_bpp: 0.016262
    stage3_full_stream_bpp: 0.028591
  CLIC professional valid:
    run: 20260627_stage3_residual_poshuff_d32_b4_r025_4096calib_clicval64_eval_compacthdr_crc32_shortid
    wandb: wandb/offline-run-20260627_130118-b90yfyj5
    total_payload_bpp: 0.015172
    stage3_full_stream_bpp: 0.027502

roundtrip_failure_count: 0 for all short-ID compact CRC32 runs
```

## Compact v3 Container Recheck

The Stage 3 position-Huffman payloads were re-evaluated with the compact v3
container. This keeps the same semantic/detail payloads and reconstructed
images, but replaces fixed-width header fields and literal strings with varints
and small table IDs. The evaluation still counts exact transmitted
`CoSERBitstream` bytes after pack/unpack roundtrip.

Active d32 b5 r0.25 position Huffman, compact v3 CRC32:

```text
Kodak 24:
  run: 20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_kodak_eval_compactv3_crc32_shortid
  wandb: wandb/offline-run-20260627_131040-5rsh6r1j
  total_payload_bpp: 0.018971761067708332
  stage3_full_stream_bpp: 0.023976643880208332
  previous compact-v2 CRC32 short-ID full_stream_bpp: 0.031300862630208336
  PSNR delta vs semantic-only: +0.3957391579945906 dB
  MS-SSIM delta vs semantic-only: +0.005756633977095249

DIV2K first 100:
  run: 20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_div2k100_eval_compactv3_crc32_shortid
  wandb: wandb/offline-run-20260627_131119-i941bt45
  total_payload_bpp: 0.018973388671875
  stage3_full_stream_bpp: 0.0239794921875
  PSNR delta vs semantic-only: +0.4601301479339597 dB
  MS-SSIM delta vs semantic-only: +0.005369908213615382

CLIC professional valid, 41 usable crops:
  run: 20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_clicproval64_eval_compactv3_crc32_shortid
  wandb: wandb/offline-run-20260627_131947-7dho224l
  total_payload_bpp: 0.017872880144817072
  stage3_full_stream_bpp: 0.022877762957317072
  previous compact-v2 CRC32 short-ID full_stream_bpp: 0.030201981707317072
  PSNR delta vs semantic-only: +0.7170708819133509 dB
  MS-SSIM delta vs semantic-only: +0.003696853794702637

CLIC professional+mobile valid, first 64:
  run: 20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_clicval64_eval_compactv3_crc32_shortid
  wandb: wandb/offline-run-20260627_131151-pbzv3rnm
  total_payload_bpp: 0.018075942993164062
  stage3_full_stream_bpp: 0.023080825805664062
  PSNR delta vs semantic-only: +0.6828677207231522 dB
  MS-SSIM delta vs semantic-only: +0.003980446141213179
```

Low-rate d32 b4 r0.25 position Huffman, compact v3 CRC32:

```text
Kodak 24:
  run: 20260627_stage3_residual_poshuff_d32_b4_r025_4096calib_kodak_eval_compactv3_crc32_shortid
  wandb: wandb/offline-run-20260627_131212-mk0oi63a
  total_payload_bpp: 0.016204833984375
  stage3_full_stream_bpp: 0.021209716796875
  previous compact-v2 CRC32 short-ID full_stream_bpp: 0.028533935546875
  PSNR delta vs semantic-only: +0.37052687009175855 dB
  MS-SSIM delta vs semantic-only: +0.004654442270596859

DIV2K first 100:
  run: 20260627_stage3_residual_poshuff_d32_b4_r025_4096calib_div2k100_eval_compactv3_crc32_shortid
  wandb: wandb/offline-run-20260627_131251-pfi3wdwy
  total_payload_bpp: 0.01626220703125
  stage3_full_stream_bpp: 0.02126708984375
  PSNR delta vs semantic-only: +0.30548624038696204 dB
  MS-SSIM delta vs semantic-only: +0.004026736319065027

CLIC professional valid, 41 usable crops:
  run: 20260627_stage3_residual_poshuff_d32_b4_r025_4096calib_clicproval64_eval_compactv3_crc32_shortid
  wandb: wandb/offline-run-20260627_132013-sutp1qr1
  total_payload_bpp: 0.015172446646341464
  stage3_full_stream_bpp: 0.020177329458841462
  previous compact-v2 CRC32 short-ID full_stream_bpp: 0.027501548208841462
  PSNR delta vs semantic-only: +0.5648934899306859 dB
  MS-SSIM delta vs semantic-only: +0.002462687288842469

CLIC professional+mobile valid, first 64:
  run: 20260627_stage3_residual_poshuff_d32_b4_r025_4096calib_clicval64_eval_compactv3_crc32_shortid
  wandb: wandb/offline-run-20260627_131325-2nf8ndjf
  total_payload_bpp: 0.015399932861328125
  stage3_full_stream_bpp: 0.020404815673828125
  PSNR delta vs semantic-only: +0.5438231080770493 dB
  MS-SSIM delta vs semantic-only: +0.0025759688578546047
```

All six compact v3 Stage 3 runs have exact semantic token and detail code
roundtrip with `roundtrip_failure_count: 0`.

## Cross-Dataset Residual Grid Diagnostics

After compact v3 evaluation, residual-grid diagnostics were extended beyond
Kodak to check whether the active range/bit choices are dataset-specific.

```text
DIV2K first 100, d32 b5 r0.25:
  run: 20260627_stage3_residual_grid_diag_d32_b5_r025_div2k100
  wandb: wandb/offline-run-20260627_132259-48qwlt2j
  symbol_entropy_bits: 2.885082
  mean_huffman_bits_per_symbol: 2.900781
  position_entropy_mean: 2.758890
  global_abs_mean: 0.021706
  global_rms: 0.030055
  global_clip_ratio: 0.0

DIV2K first 100, d32 b4 r0.25:
  run: 20260627_stage3_residual_grid_diag_d32_b4_r025_div2k100
  wandb: wandb/offline-run-20260627_132315-c3s8xr0p
  symbol_entropy_bits: 1.930494
  mean_huffman_bits_per_symbol: 1.975156
  position_entropy_mean: 1.861790
  global_abs_mean: 0.021706
  global_rms: 0.030056
  global_clip_ratio: 0.0

CLIC professional valid, 41 usable crops, d32 b5 r0.25:
  run: 20260627_stage3_residual_grid_diag_d32_b5_r025_clicproval41
  wandb: wandb/offline-run-20260627_132331-32a03c6l
  symbol_entropy_bits: 2.653491
  mean_huffman_bits_per_symbol: 2.701982
  position_entropy_mean: 2.449102
  global_abs_mean: 0.018461
  global_rms: 0.024644
  global_clip_ratio: 0.0

CLIC professional valid, 41 usable crops, d32 b4 r0.25:
  run: 20260627_stage3_residual_grid_diag_d32_b4_r025_clicproval41
  wandb: wandb/offline-run-20260627_132346-zfg7cib3
  symbol_entropy_bits: 1.712887
  mean_huffman_bits_per_symbol: 1.779599
  position_entropy_mean: 1.600201
  global_abs_mean: 0.018461
  global_rms: 0.024643
  global_clip_ratio: 0.0
```

Interpretation: range 0.25 is not clipping on Kodak, DIV2K, or CLIC. DIV2K has
the widest residual distribution among the three checked datasets, explaining
its slightly higher detail payload. CLIC professional valid has lower residual
energy and entropy, so its lower detail bpp is expected. The next learned
detail model should prioritize conditional entropy modeling and spatially
adaptive residual allocation rather than increasing the scalar range.

## 8192-Crop Residual Prior Calibration

The active position-Huffman residual priors were refit with the same
OpenImages+DIV2K calibration roots as the previous 4096-crop priors, but with
8192 crops. A mistaken probe using the smaller `race_pilot_*` roots produced
only 256 calibration crops and is not active.

```text
active b5 prior:
  run: 20260627_stage3_residual_poshuff_d32_b5_r025_8192calib_openimages_div2k
  wandb: wandb/offline-run-20260627_132619-hb790ygs
  prior: outputs/stage3_residual_entropy/20260627_stage3_residual_poshuff_d32_b5_r025_8192calib_openimages_div2k/static_residual_grid_position_huffman_prior.json
  num_images: 8192
  symbol_entropy_bits: 2.892438
  mean_huffman_bits_per_symbol: 2.896092
  calibration_clip_ratio: 0.000156

active low-rate b4 prior:
  run: 20260627_stage3_residual_poshuff_d32_b4_r025_8192calib_openimages_div2k
  wandb: wandb/offline-run-20260627_132854-lb5a7r8q
  prior: outputs/stage3_residual_entropy/20260627_stage3_residual_poshuff_d32_b4_r025_8192calib_openimages_div2k/static_residual_grid_position_huffman_prior.json
  num_images: 8192
  symbol_entropy_bits: 1.932172
  mean_huffman_bits_per_symbol: 1.961296
  calibration_clip_ratio: 0.000156
```

Actual compact-v3 evaluation against the 4096-crop active anchors:

```text
b5 Kodak 24:
  8192 total_payload_bpp: 0.0189666748046875
  8192 full_stream_bpp: 0.0239715576171875
  previous 4096 full_stream_bpp: 0.023976643880208332

b5 DIV2K first 100:
  8192 total_payload_bpp: 0.018973388671875
  8192 full_stream_bpp: 0.0239794921875
  previous 4096 full_stream_bpp: 0.0239794921875

b5 CLIC professional valid, 41 usable crops:
  8192 total_payload_bpp: 0.017872880144817072
  8192 full_stream_bpp: 0.022877762957317072
  previous 4096 full_stream_bpp: 0.022877762957317072

b4 Kodak 24:
  8192 total_payload_bpp: 0.016199747721354168
  8192 full_stream_bpp: 0.021204630533854168
  previous 4096 full_stream_bpp: 0.021209716796875

b4 DIV2K first 100:
  8192 total_payload_bpp: 0.016260986328125
  8192 full_stream_bpp: 0.021265869140625
  previous 4096 full_stream_bpp: 0.02126708984375

b4 CLIC professional valid, 41 usable crops:
  8192 total_payload_bpp: 0.01516649199695122
  8192 full_stream_bpp: 0.02017137480945122
  previous 4096 full_stream_bpp: 0.020177329458841462
```

Decision: promote the 8192-crop residual priors because they are non-regressive
and slightly better on Kodak/b4 CLIC, but the gain is too small to justify more
manual effort on static Huffman calibration. Future bitrate gains should come
from learned conditional residual entropy, not larger static calibration.

## Learned Residual AE Probe

A compact semantic-conditioned residual autoencoder was implemented as a
candidate replacement for the fixed residual grid:

```text
model: src/coserdic/models/residual_detail.py
train: scripts/train_stage3_residual_autoencoder.py
eval: scripts/eval_stage3_residual_autoencoder_bitstream.py
latent shape: 3 x 8 x 8 for 256 x 256 crops
quantizer: 5-bit uniform scalar quantizer, range 0.25
bitstream: semantic topk/escape payload + learned detail static-Huffman payload
```

The first AMP smoke run produced non-finite gradients, so the probe was run in
FP32. The encoder output was range-bounded with `tanh` and zero-initialized so
training starts close to the semantic-only reconstruction instead of immediately
clipping the detail latent.

Actual Kodak bitstream evaluation:

```text
500step no-rate:
  train run: 20260627_stage3_residual_ae_d32_c3_b5_r025_500step_norate_fp32
  eval run: 20260627_stage3_residual_ae_d32_c3_b5_r025_500step_norate_kodak_eval
  wandb eval: wandb/offline-run-20260627_120723-i0hobi97
  detail_payload_bpp: 0.006577
  total_payload_bpp: 0.017298
  full_stream_bpp: 0.077382
  PSNR delta: +0.0088 dB
  MS-SSIM delta: +0.00008
  roundtrip_failure_count: 0

2500step no-rate from 500step:
  train run: 20260627_stage3_residual_ae_d32_c3_b5_r025_2500step_norate_from500_fp32
  eval run: 20260627_stage3_residual_ae_d32_c3_b5_r025_2500step_norate_kodak_eval
  wandb eval: wandb/offline-run-20260627_121413-u0c101cu
  detail_payload_bpp: 0.014648
  total_payload_bpp: 0.025370
  full_stream_bpp: 0.085576
  PSNR delta: +0.2073 dB
  MS-SSIM delta: +0.01159
  detail entropy: 4.84 bits/symbol
  roundtrip_failure_count: 0

1000step rate-proxy 0.03 from 2500step:
  train run: 20260627_stage3_residual_ae_d32_c3_b5_r025_1000step_rate003_from2500_fp32
  eval run: 20260627_stage3_residual_ae_d32_c3_b5_r025_1000step_rate003_from2500_kodak_eval
  wandb eval: wandb/offline-run-20260627_121705-wkpmlk0f
  detail_payload_bpp: 0.004344
  total_payload_bpp: 0.015066
  full_stream_bpp: 0.075150
  PSNR delta: +0.0112 dB
  MS-SSIM delta: +0.00007
  detail entropy: 0.99 bits/symbol
  roundtrip_failure_count: 0
```

Interpretation: the learned residual AE path is functional and uses actual
transmitted bytes end-to-end, but it does not yet dominate the fixed residual
grid. Without rate pressure it spends too many bits; with the current rate proxy
it collapses to an almost zero-detail two-symbol stream. Keep it as an
implementation scaffold for the next learned detail model, not as the active
Core MVP Stage 3 codec.

## Interpretation

The fixed residual grid is not a final model, but it proves that a transmitted
detail stream can improve fidelity while keeping the semantic/detail split
clean and auditable. The static-Huffman update is especially important because
it keeps the reconstruction identical to zlib-fixed d32/b4/r0.25 while reducing
Kodak detail payload from 0.008759 bpp to 0.005544 bpp.

The most useful signal is that an 8x8 RGB residual grid at range 0.25 improves
all three checked datasets. The active anchors are now semantic-position-left
Huffman with smoothing=0: g8 for the d32/b4/r0.25 low-rate setting and g4 for
the d32/b5/r0.25 quality setting. They preserve the same decoded residual
grids while reducing actual transmitted bytes. The residual-grid clipping ratio
is zero on Kodak, and the code entropy is far below the fixed width. This
suggests Stage 3 should start simple:

```text
semantic-conditioned residual prediction
small latent grid
learned entropy model
actual bytes only
```

The d16/b4/r0.25 setting is a high-rate oracle:

```text
Kodak detail_payload_bpp: 0.031204
PSNR delta: +0.8644 dB
MS-SSIM delta: +0.03481
```

It is useful as an upper diagnostic, but not the low-bitrate active bootstrap.

## Next

```text
1. Extend residual distribution diagnostics for d32/b4/r0.25 with per-channel
   histograms and spatial energy maps.
2. Refine the learned residual AE with a real entropy objective or a simpler
   teacher-distilled residual predictor instead of the current absolute-latent
   proxy.
3. Evaluate with actual_payload_bpp and debug_full_stream_bpp, not estimated bpp.
4. Keep d32/b5/r0.25 semantic-position-left g4 Huffman with smoothing=0 as the
   quality bootstrap and d32/b4/r0.25 semantic-position-left g4 Huffman with
   smoothing=0 as the low-rate anchor.
```

## Semantic-Conditioned Residual Huffman Probe

Implemented and evaluated a semantic-position residual Huffman codec. The
decoded residual grid is unchanged; only the detail entropy table is conditioned
on the decoded semantic token group and detail position.

```text
codec: StaticResidualGridSemanticPositionHuffmanCode
implementation: src/coserdic/entropy/residual_grid.py
fit script: scripts/fit_stage3_residual_huffman_prior.py
eval script: scripts/eval_stage3_uniform_residual_bitstream.py
unit tests: tests/test_residual_grid.py, tests/test_bitstream.py
actual stream: compact-v3 CoSERBitstream + CRC32
```

Calibration on 8192 OpenImages+DIV2K crops:

```text
g2 mean_huffman_bits_per_symbol: 1.940376
g4 mean_huffman_bits_per_symbol: 1.930738
g8 mean_huffman_bits_per_symbol: 1.925484
g16 mean_huffman_bits_per_symbol: 1.926224
```

Actual-byte low-rate comparison against the d32/b4/r0.25 position-Huffman
anchor:

```text
position-Huffman b4 anchor:
  Kodak full_stream_bpp: 0.021205
  DIV2K100 full_stream_bpp: 0.021266
  CLIC professional valid 41 full_stream_bpp: 0.020171
  CLIC professional+mobile valid 64 full_stream_bpp: 0.020405

semantic-position g4:
  Kodak:
    run: 20260627_stage3_residual_semposhuff_g4_d32_b4_r025_8192calib_kodak_eval_compactv3_crc32
    wandb: wandb/offline-run-20260627_134406-ywxky0zy
    detail_payload_bpp: 0.005493
    full_stream_bpp: 0.021220
    delta vs position b4: +0.000015
  DIV2K100:
    run: 20260627_stage3_residual_semposhuff_g4_d32_b4_r025_8192calib_div2k100_eval_compactv3_crc32
    wandb: wandb/offline-run-20260627_134653-eao2bile
    detail_payload_bpp: 0.005690
    full_stream_bpp: 0.021119
    delta vs position b4: -0.000146
  CLIC professional valid 41:
    run: 20260627_stage3_residual_semposhuff_g4_d32_b4_r025_8192calib_clicpro41_eval_compactv3_crc32
    wandb: wandb/offline-run-20260627_134724-uk0ynpy0
    detail_payload_bpp: 0.005207
    full_stream_bpp: 0.020118
    delta vs position b4: -0.000054
  CLIC professional+mobile valid 64:
    run: 20260627_stage3_residual_semposhuff_g4_d32_b4_r025_8192calib_clicval64_eval_compactv3_crc32
    wandb: wandb/offline-run-20260627_134821-8xe59otf
    detail_payload_bpp: 0.005255
    full_stream_bpp: 0.020309
    delta vs position b4: -0.000096

semantic-position g8:
  Kodak full_stream_bpp: 0.021245
  DIV2K100 full_stream_bpp: 0.021101
  CLIC professional valid 41 full_stream_bpp: 0.020118
```

All semantic and detail roundtrips were exact. g4 is the best cross-domain
candidate because it improves DIV2K/CLIC actual bpp while keeping the Kodak
regression very small. g8 is better on DIV2K but less Kodak-safe. g2 and g16
are not useful enough to continue.

Intermediate decision before the hybrid selector: keep the d32/b5/r0.25
position-Huffman codec as the quality bootstrap and keep the d32/b4/r0.25
position-Huffman codec as the headline Kodak-safe low-rate anchor. Track
semantic-position g4 as the cross-domain low-rate candidate and revisit it with
smoothing sweeps or an explicit transmitted hybrid mode selector.

## Hybrid Position/Semantic-Position Low-Rate Anchor

Implemented a hybrid residual entropy codec that transmits one explicit mode bit
per image:

```text
0: d32/b4/r0.25 position Huffman
1: d32/b4/r0.25 semantic-position g4 Huffman
selection rule: minimum actual payload byte length after adding the mode bit
prior: outputs/stage3_residual_entropy/20260627_stage3_residual_hybrid_pos_semposg4_d32_b4_r025_8192calib_openimages_div2k/static_residual_grid_hybrid_huffman_prior.json
```

Actual compact-v3 CRC32 full-stream results:

```text
Kodak:
  run: 20260627_stage3_residual_hybrid_pos_semposg4_d32_b4_r025_8192calib_kodak_eval_compactv3_crc32
  wandb: wandb/offline-run-20260627_135730-27tc4yrq
  detail_payload_bpp: 0.005432
  total_payload_bpp: 0.016154
  full_stream_bpp: 0.021159
  previous position b4 full_stream_bpp: 0.021205
  semantic-position mode rate: 0.5000

DIV2K first 100:
  run: 20260627_stage3_residual_hybrid_pos_semposg4_d32_b4_r025_8192calib_div2k100_eval_compactv3_crc32
  wandb: wandb/offline-run-20260627_135822-dwqeioda
  detail_payload_bpp: 0.005649
  total_payload_bpp: 0.016074
  full_stream_bpp: 0.021079
  previous position b4 full_stream_bpp: 0.021266
  semantic-position mode rate: 0.6500

CLIC professional valid 41:
  run: 20260627_stage3_residual_hybrid_pos_semposg4_d32_b4_r025_8192calib_clicpro41_eval_compactv3_crc32
  wandb: wandb/offline-run-20260627_135857-e8675327
  detail_payload_bpp: 0.005106
  total_payload_bpp: 0.015012
  full_stream_bpp: 0.020017
  previous position b4 full_stream_bpp: 0.020171
  semantic-position mode rate: 0.6341

CLIC professional+mobile valid 64:
  run: 20260627_stage3_residual_hybrid_pos_semposg4_d32_b4_r025_8192calib_clicval64_eval_compactv3_crc32
  wandb: wandb/offline-run-20260627_135936-m93cc9h7
  detail_payload_bpp: 0.005178
  total_payload_bpp: 0.015228
  full_stream_bpp: 0.020233
  previous position b4 full_stream_bpp: 0.020405
  semantic-position mode rate: 0.6875
```

All semantic/detail roundtrips were exact. Intermediate decision: promote hybrid
position/semantic-position g4 as the active d32/b4/r0.25 low-rate anchor. Keep
d32/b5/r0.25 position Huffman as the quality bootstrap until the b5 hybrid
follow-up below.

Follow-up: g8 semantic-position hybrid was evaluated as an upper check and
supersedes g4 for the active low-rate anchor.

```text
g8 hybrid prior:
  outputs/stage3_residual_entropy/20260627_stage3_residual_hybrid_pos_semposg8_d32_b4_r025_8192calib_openimages_div2k/static_residual_grid_hybrid_huffman_prior.json

Kodak:
  run: 20260627_stage3_residual_hybrid_pos_semposg8_d32_b4_r025_8192calib_kodak_eval_compactv3_crc32
  wandb: wandb/offline-run-20260627_140417-r6l4r9qq
  detail_payload_bpp: 0.005412
  total_payload_bpp: 0.016134
  full_stream_bpp: 0.021139
  semantic-position mode rate: 0.4167

DIV2K first 100:
  run: 20260627_stage3_residual_hybrid_pos_semposg8_d32_b4_r025_8192calib_div2k100_eval_compactv3_crc32
  wandb: wandb/offline-run-20260627_140517-ryfeudpe
  detail_payload_bpp: 0.005627
  total_payload_bpp: 0.016052
  full_stream_bpp: 0.021057
  semantic-position mode rate: 0.6800

CLIC professional valid 41:
  run: 20260627_stage3_residual_hybrid_pos_semposg8_d32_b4_r025_8192calib_clicpro41_eval_compactv3_crc32
  wandb: wandb/offline-run-20260627_140549-690vmy0n
  detail_payload_bpp: 0.005085
  total_payload_bpp: 0.014991
  full_stream_bpp: 0.019996
  semantic-position mode rate: 0.6098

CLIC professional+mobile valid 64:
  run: 20260627_stage3_residual_hybrid_pos_semposg8_d32_b4_r025_8192calib_clicval64_eval_compactv3_crc32
  wandb: wandb/offline-run-20260627_140633-xae1dhzu
  detail_payload_bpp: 0.005180
  total_payload_bpp: 0.015230
  full_stream_bpp: 0.020235
  semantic-position mode rate: 0.6563
```

Decision update: promote hybrid position/semantic-position g8 as the active
d32/b4/r0.25 low-rate anchor. g4 hybrid remains a slightly better CLIC64-only
variant, but g8 wins the broader Kodak/DIV2K/CLIC41 comparison.

## Hybrid Position/Semantic-Position Quality Bootstrap

The same transmitted-mode hybrid selector was applied to the d32/b5/r0.25
quality bootstrap. This is an entropy-only change: the decoded b5 residual grid
and quality metrics are unchanged relative to the b5 position-Huffman anchor.

```text
g8 hybrid b5 prior:
  outputs/stage3_residual_entropy/20260627_stage3_residual_hybrid_pos_semposg8_d32_b5_r025_8192calib_openimages_div2k/static_residual_grid_hybrid_huffman_prior.json

Kodak:
  run: 20260627_stage3_residual_hybrid_pos_semposg8_d32_b5_r025_8192calib_kodak_eval_compactv3_crc32
  wandb: wandb/offline-run-20260627_141219-g9wkvgad
  detail_payload_bpp: 0.008169
  total_payload_bpp: 0.018890
  full_stream_bpp: 0.023895
  previous position b5 full_stream_bpp: 0.023972
  PSNR delta: +0.3957 dB
  MS-SSIM delta: +0.00576
  semantic-position mode rate: 0.6250

DIV2K first 100:
  run: 20260627_stage3_residual_hybrid_pos_semposg8_d32_b5_r025_8192calib_div2k100_eval_compactv3_crc32
  wandb: wandb/offline-run-20260627_141313-ns4qugzp
  detail_payload_bpp: 0.008427
  total_payload_bpp: 0.018851
  full_stream_bpp: 0.023856
  previous position b5 full_stream_bpp: 0.023979
  PSNR delta: +0.4601 dB
  MS-SSIM delta: +0.00537
  semantic-position mode rate: 0.6100

CLIC professional valid 41:
  run: 20260627_stage3_residual_hybrid_pos_semposg8_d32_b5_r025_8192calib_clicpro41_eval_compactv3_crc32
  wandb: wandb/offline-run-20260627_141346-kg55a39a
  detail_payload_bpp: 0.007848
  total_payload_bpp: 0.017754
  full_stream_bpp: 0.022759
  previous position b5 full_stream_bpp: 0.022878
  PSNR delta: +0.7171 dB
  MS-SSIM delta: +0.00370
  semantic-position mode rate: 0.5854

CLIC professional+mobile valid 64:
  run: 20260627_stage3_residual_hybrid_pos_semposg8_d32_b5_r025_8192calib_clicval64_eval_compactv3_crc32
  wandb: wandb/offline-run-20260627_141615-2y7fctbv
  detail_payload_bpp: 0.007906
  total_payload_bpp: 0.017956
  full_stream_bpp: 0.022961
  previous position b5 full_stream_bpp: 0.023081
  PSNR delta: +0.6829 dB
  MS-SSIM delta: +0.00398
  semantic-position mode rate: 0.6094
```

Intermediate decision: g8 hybrid beats pure position Huffman for the
d32/b5/r0.25 quality bootstrap. The previous b5 position-Huffman prior remains
the ablation anchor, but it is no longer the active setting.

## B5 Hybrid Group-Count Check

Because the best b4 low-rate group count does not have to be the best b5
quality group count, the same b5 hybrid evaluation was repeated with a g4
semantic-position prior.

```text
g4 fit:
  run: 20260627_stage3_residual_semposhuff_g4_d32_b5_r025_8192calib_openimages_div2k
  wandb: wandb/offline-run-20260627_142232-t27ij90u
  mean_huffman_bits_per_symbol: 2.864717
  g8 mean_huffman_bits_per_symbol: 2.855923

g4 hybrid prior:
  outputs/stage3_residual_entropy/20260627_stage3_residual_hybrid_pos_semposg4_d32_b5_r025_8192calib_openimages_div2k/static_residual_grid_hybrid_huffman_prior.json
```

Actual compact-v3 CRC32 full-stream results:

```text
Kodak:
  run: 20260627_stage3_residual_hybrid_pos_semposg4_d32_b5_r025_8192calib_kodak_eval_compactv3_crc32
  wandb: wandb/offline-run-20260627_142315-6xlha7gw
  detail_payload_bpp: 0.008158
  total_payload_bpp: 0.018880
  full_stream_bpp: 0.023885
  g8 hybrid full_stream_bpp: 0.023895
  previous position b5 full_stream_bpp: 0.023972
  semantic-position mode rate: 0.5000

DIV2K first 100:
  run: 20260627_stage3_residual_hybrid_pos_semposg4_d32_b5_r025_8192calib_div2k100_eval_compactv3_crc32
  wandb: wandb/offline-run-20260627_142406-cet0880n
  detail_payload_bpp: 0.008436
  total_payload_bpp: 0.018861
  full_stream_bpp: 0.023866
  g8 hybrid full_stream_bpp: 0.023856
  previous position b5 full_stream_bpp: 0.023979
  semantic-position mode rate: 0.6400

CLIC professional valid 41:
  run: 20260627_stage3_residual_hybrid_pos_semposg4_d32_b5_r025_8192calib_clicpro41_eval_compactv3_crc32
  wandb: wandb/offline-run-20260627_142439-t0dmfi53
  detail_payload_bpp: 0.007842
  total_payload_bpp: 0.017748
  full_stream_bpp: 0.022753
  g8 hybrid full_stream_bpp: 0.022759
  previous position b5 full_stream_bpp: 0.022878
  semantic-position mode rate: 0.6829

CLIC professional+mobile valid 64:
  run: 20260627_stage3_residual_hybrid_pos_semposg4_d32_b5_r025_8192calib_clicval64_eval_compactv3_crc32
  wandb: wandb/offline-run-20260627_142520-cpvoz6vf
  detail_payload_bpp: 0.007904
  total_payload_bpp: 0.017954
  full_stream_bpp: 0.022959
  g8 hybrid full_stream_bpp: 0.022961
  previous position b5 full_stream_bpp: 0.023081
  semantic-position mode rate: 0.7188
```

Intermediate decision before the smoothing check: promote hybrid
position/semantic-position g4 as the d32/b5/r0.25 quality bootstrap. g4 wins
Kodak, CLIC professional valid 41, and CLIC professional+mobile valid 64; g8
wins DIV2K100 by only 0.000010 bpp. The four-dataset mean favors g4 by about
0.000002 bpp. Keep d32/b4/r0.25 low-rate on g8.

## B5 G4 Smoothing Check

The active b5 g4 semantic-position prior was then refit with
`smoothing_count=0`. This increases maximum code length to 17, so the check was
kept actual-byte and cross-dataset rather than using calibration bits alone.

```text
g4 smoothing=0 fit:
  run: 20260627_stage3_residual_semposhuff_g4_sm0_d32_b5_r025_8192calib_openimages_div2k
  wandb: wandb/offline-run-20260627_143124-sa9qq3lv
  mean_huffman_bits_per_symbol: 2.855390
  previous g4 smoothing=1 mean_huffman_bits_per_symbol: 2.864717

g4 smoothing=0 hybrid prior:
  outputs/stage3_residual_entropy/20260627_stage3_residual_hybrid_pos_semposg4_sm0_d32_b5_r025_8192calib_openimages_div2k/static_residual_grid_hybrid_huffman_prior.json
```

Actual compact-v3 CRC32 full-stream results:

```text
Kodak:
  run: 20260627_stage3_residual_hybrid_pos_semposg4_sm0_d32_b5_r025_8192calib_kodak_eval_compactv3_crc32
  wandb: wandb/offline-run-20260627_143205-fdb9h7of
  detail_payload_bpp: 0.008153
  total_payload_bpp: 0.018875
  full_stream_bpp: 0.023880
  previous g4 smoothing=1 full_stream_bpp: 0.023885
  previous g8 hybrid full_stream_bpp: 0.023895
  previous position b5 full_stream_bpp: 0.023972
  semantic-position mode rate: 0.6250

DIV2K first 100:
  run: 20260627_stage3_residual_hybrid_pos_semposg4_sm0_d32_b5_r025_8192calib_div2k100_eval_compactv3_crc32
  wandb: wandb/offline-run-20260627_143256-vskufx4d
  detail_payload_bpp: 0.008428
  total_payload_bpp: 0.018853
  full_stream_bpp: 0.023857
  previous g4 smoothing=1 full_stream_bpp: 0.023866
  previous g8 hybrid full_stream_bpp: 0.023856
  previous position b5 full_stream_bpp: 0.023979
  semantic-position mode rate: 0.7100

CLIC professional valid 41:
  run: 20260627_stage3_residual_hybrid_pos_semposg4_sm0_d32_b5_r025_8192calib_clicpro41_eval_compactv3_crc32
  wandb: wandb/offline-run-20260627_143332-rwwvp714
  detail_payload_bpp: 0.007821
  total_payload_bpp: 0.017727
  full_stream_bpp: 0.022732
  previous g4 smoothing=1 full_stream_bpp: 0.022753
  previous g8 hybrid full_stream_bpp: 0.022759
  previous position b5 full_stream_bpp: 0.022878
  semantic-position mode rate: 0.7561

CLIC professional+mobile valid 64:
  run: 20260627_stage3_residual_hybrid_pos_semposg4_sm0_d32_b5_r025_8192calib_clicval64_eval_compactv3_crc32
  wandb: wandb/offline-run-20260627_143414-ejmfd49b
  detail_payload_bpp: 0.007881
  total_payload_bpp: 0.017931
  full_stream_bpp: 0.022936
  previous g4 smoothing=1 full_stream_bpp: 0.022959
  previous g8 hybrid full_stream_bpp: 0.022961
  previous position b5 full_stream_bpp: 0.023081
  semantic-position mode rate: 0.7656
```

Decision update: promote b5 hybrid position/semantic-position g4 with
smoothing=0 as the active quality bootstrap. It improves all checked datasets
relative to g4 smoothing=1, and all roundtrips remain exact. DIV2K100 remains
0.000001 bpp behind g8 hybrid, but the four-dataset mean and CLIC robustness
favor g4 smoothing=0.

## B4 G8 Smoothing Check

The same no-smoothing check was applied to the active b4 low-rate g8 prior.

```text
g8 smoothing=0 fit:
  run: 20260627_stage3_residual_semposhuff_g8_sm0_d32_b4_r025_8192calib_openimages_div2k
  wandb: wandb/offline-run-20260627_144002-pnzr0xam
  mean_huffman_bits_per_symbol: 1.917438
  previous g8 smoothing=1 mean_huffman_bits_per_symbol: 1.925484

g8 smoothing=0 hybrid prior:
  outputs/stage3_residual_entropy/20260627_stage3_residual_hybrid_pos_semposg8_sm0_d32_b4_r025_8192calib_openimages_div2k/static_residual_grid_hybrid_huffman_prior.json
```

Actual compact-v3 CRC32 full-stream results:

```text
Kodak:
  run: 20260627_stage3_residual_hybrid_pos_semposg8_sm0_d32_b4_r025_8192calib_kodak_eval_compactv3_crc32
  wandb: wandb/offline-run-20260627_144127-32qhmh9y
  detail_payload_bpp: 0.005402
  total_payload_bpp: 0.016123
  full_stream_bpp: 0.021128
  previous g8 smoothing=1 full_stream_bpp: 0.021139

DIV2K first 100:
  run: 20260627_stage3_residual_hybrid_pos_semposg8_sm0_d32_b4_r025_8192calib_div2k100_eval_compactv3_crc32
  wandb: wandb/offline-run-20260627_144219-l5o0nqw0
  detail_payload_bpp: 0.005609
  total_payload_bpp: 0.016034
  full_stream_bpp: 0.021039
  previous g8 smoothing=1 full_stream_bpp: 0.021057

CLIC professional valid 41:
  run: 20260627_stage3_residual_hybrid_pos_semposg8_sm0_d32_b4_r025_8192calib_clicpro41_eval_compactv3_crc32
  wandb: wandb/offline-run-20260627_144251-0pllsojt
  detail_payload_bpp: 0.005079
  total_payload_bpp: 0.014985
  full_stream_bpp: 0.019990
  previous g8 smoothing=1 full_stream_bpp: 0.019996

CLIC professional+mobile valid 64:
  run: 20260627_stage3_residual_hybrid_pos_semposg8_sm0_d32_b4_r025_8192calib_clicval64_eval_compactv3_crc32
  wandb: wandb/offline-run-20260627_144331-boc32s96
  detail_payload_bpp: 0.005171
  total_payload_bpp: 0.015221
  full_stream_bpp: 0.020226
  previous g8 smoothing=1 full_stream_bpp: 0.020235
```

Decision update: promote b4 hybrid position/semantic-position g8 with
smoothing=0 as the active low-rate anchor. It improves all four checked
datasets and preserves exact semantic/detail roundtrip.

## Bpp Reporting Policy Alignment

Decision 010 reclassifies the main LIC paper metric as `actual_payload_bpp`,
matching CompressAI-style entropy-coded payload byte accounting. For current
Stage3 uniform residual runs:

```text
actual_payload_bpp = paper_bpp = semantic_payload_bpp + detail_payload_bpp
debug_full_stream_bpp = compact-v3 CoSERBitstream bytes including header/checksum
```

The previous `stage3_full_stream_bpp` values remain useful for roundtrip and
container-overhead audits, but they are no longer the main paper comparison
bpp. Future eval runs emit explicit `actual_payload_bpp_mean`,
`paper_bpp_mean`, and `debug_full_stream_bpp_mean` fields while preserving
legacy `total_payload_bpp_mean` and `stage3_full_stream_bpp_mean` aliases.

Previous paper-style anchors before semantic-position-left update:

```text
b5 quality anchor, hybrid g4 smoothing=0:
  Kodak actual_payload_bpp: 0.018875
  DIV2K100 actual_payload_bpp: 0.018853
  CLIC professional valid 41 actual_payload_bpp: 0.017727
  CLIC professional+mobile valid 64 actual_payload_bpp: 0.017931

b4 low-rate anchor, hybrid g8 smoothing=0:
  Kodak actual_payload_bpp: 0.016123
  DIV2K100 actual_payload_bpp: 0.016034
  CLIC professional valid 41 actual_payload_bpp: 0.014985
  CLIC professional+mobile valid 64 actual_payload_bpp: 0.015221
```

## Semantic-Position-Left Context Huffman

After aligning the bpp policy, Stage 3 entropy coding was tightened by adding a
causal left-residual context to the semantic-position Huffman prior. This does
not transmit any per-image table or mask. The decoder already knows the
semantic tokens and decodes residual codes in channel-major raster order, so the
left detail code is available when each next residual symbol is decoded.

Implementation:

```text
codec:
  StaticResidualGridSemanticPositionLeftContextHuffmanCode
  src/coserdic/entropy/residual_grid.py

fit:
  scripts/fit_stage3_residual_huffman_prior.py
  --coding-mode semantic_position_leftctx_huffman

eval:
  scripts/eval_stage3_uniform_residual_bitstream.py
  --detail-codec semantic_position_leftctx_huffman
```

Calibration:

```text
b4 low-rate:
  run: 20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_openimages_div2k
  wandb: wandb/offline-run-20260627_153936-20g6oib6
  prior:
    outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_openimages_div2k/static_residual_grid_semantic_position_leftctx_huffman_prior.json
  mean_huffman_bits_per_symbol: 1.788819
  previous semantic-position-left g8 smoothing=0: 1.779746
  previous semantic-position g8 smoothing=0: 1.917438

b5 quality:
  run: 20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_8192calib_openimages_div2k
  wandb: wandb/offline-run-20260627_152106-v74n0yot
  prior:
    outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_8192calib_openimages_div2k/static_residual_grid_semantic_position_leftctx_huffman_prior.json
  mean_huffman_bits_per_symbol: 2.697383
  previous semantic-position g4 smoothing=0: 2.855390
```

Actual-payload results:

```text
b4 low-rate, d32/b4/r0.25, semantic-position-left g4:
  Kodak:
    run: 20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_kodak_eval_compactv3_crc32
    wandb: wandb/offline-run-20260627_154002-l0fvhzxf
    detail_payload_bpp: 0.005107
    actual_payload_bpp: 0.015828
    debug_full_stream_bpp: 0.020833
    PSNR delta: +0.3705 dB
    previous active actual_payload_bpp: 0.016123

  DIV2K first 100:
    run: 20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_div2k100_eval_compactv3_crc32
    wandb: wandb/offline-run-20260627_154056-oufto3f2
    detail_payload_bpp: 0.005233
    actual_payload_bpp: 0.015658
    debug_full_stream_bpp: 0.020663
    PSNR delta: +0.3055 dB
    previous active actual_payload_bpp: 0.016034

  CLIC professional valid 41:
    run: 20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_clicpro41_eval_compactv3_crc32
    wandb: wandb/offline-run-20260627_154129-4p6l28qx
    detail_payload_bpp: 0.004802
    actual_payload_bpp: 0.014708
    debug_full_stream_bpp: 0.019713
    PSNR delta: +0.5649 dB
    previous active actual_payload_bpp: 0.014985

  CLIC professional+mobile valid 64:
    run: 20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_clicval64_eval_compactv3_crc32
    wandb: wandb/offline-run-20260627_154209-kxhgrqat
    detail_payload_bpp: 0.004881
    actual_payload_bpp: 0.014931
    debug_full_stream_bpp: 0.019936
    PSNR delta: +0.5438 dB
    previous active actual_payload_bpp: 0.015221

b5 quality, d32/b5/r0.25, semantic-position-left g4:
  Kodak:
    run: 20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_8192calib_kodak_eval_compactv3_crc32
    wandb: wandb/offline-run-20260627_152133-92lwm3b4
    detail_payload_bpp: 0.007751
    actual_payload_bpp: 0.018473
    debug_full_stream_bpp: 0.023478
    PSNR delta: +0.3957 dB
    previous active actual_payload_bpp: 0.018875

  DIV2K first 100:
    run: 20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_8192calib_div2k100_eval_compactv3_crc32
    wandb: wandb/offline-run-20260627_152426-7u7pb0ua
    detail_payload_bpp: 0.007965
    actual_payload_bpp: 0.018390
    debug_full_stream_bpp: 0.023395
    PSNR delta: +0.4601 dB
    previous active actual_payload_bpp: 0.018853

  CLIC professional valid 41:
    run: 20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_8192calib_clicpro41_eval_compactv3_crc32
    wandb: wandb/offline-run-20260627_152456-qd8fqs38
    detail_payload_bpp: 0.007351
    actual_payload_bpp: 0.017257
    debug_full_stream_bpp: 0.022261
    PSNR delta: +0.7171 dB
    previous active actual_payload_bpp: 0.017727

  CLIC professional+mobile valid 64:
    run: 20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_8192calib_clicval64_eval_compactv3_crc32
    wandb: wandb/offline-run-20260627_152534-dpm1cmqs
    detail_payload_bpp: 0.007437
    actual_payload_bpp: 0.017487
    debug_full_stream_bpp: 0.022491
    PSNR delta: +0.6829 dB
    previous active actual_payload_bpp: 0.017931
```

Decision update: promote `semantic_position_leftctx_huffman` as the active
Stage 3 residual payload codec. The b4 and b5 anchors improve all checked
datasets in `actual_payload_bpp`, preserve identical reconstruction deltas, and
have zero semantic/detail roundtrip failures.

## Semantic-Position-Left Group-Count Checks

The immediate follow-up checked whether the group counts should change after
adding the left-detail context.

```text
b5 g8 diagnostic:
  run: 20260627_stage3_residual_semposleft_g8_sm0_d32_b5_r025_8192calib_openimages_div2k
  wandb: wandb/offline-run-20260627_153522-dc1y9y7q
  mean_huffman_bits_per_symbol: 2.672975
  actual_payload_bpp:
    Kodak: 0.018489
    DIV2K100: 0.018413
    CLIC professional valid 41: 0.017257
    CLIC professional+mobile valid 64: 0.017504
  decision: keep b5 g4 active; g8 has lower calibration bits but worse actual
            payload mean after byte packing.

b4 g4 vs g8:
  g4 fit run: 20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_openimages_div2k
  g4 wandb: wandb/offline-run-20260627_153936-20g6oib6
  g4 mean_huffman_bits_per_symbol: 1.788819
  g8 mean_huffman_bits_per_symbol: 1.779746
  g4 actual_payload_bpp:
    Kodak: 0.015828
    DIV2K100: 0.015658
    CLIC professional valid 41: 0.014708
    CLIC professional+mobile valid 64: 0.014931
  g8 actual_payload_bpp:
    Kodak: 0.015828
    DIV2K100: 0.015671
    CLIC professional valid 41: 0.014702
    CLIC professional+mobile valid 64: 0.014927
  decision: use b4 g4 as active low-rate anchor because it is simpler and has
            a marginally lower four-dataset actual payload mean; keep g8 as
            the CLIC-best candidate.
```

## Per-Image Metrics Export

Stage 3 evaluation now writes a per-image JSONL next to `summary.json`:

```text
script:
  scripts/eval_stage3_uniform_residual_bitstream.py

artifact:
  per_image_metrics.jsonl

fields:
  path, payload bytes, actual_payload_bpp, debug_full_stream_bpp,
  semantic-only/stage3 PSNR, L1, MS-SSIM, roundtrip flags,
  residual-grid stats, detail code entropy
```

Smoke run:

```text
run: 20260627_stage3_per_image_jsonl_smoke_b4_g4_kodak4
wandb: wandb/offline-run-20260627_154738-qfjpz261
summary:
  results/bitstreams/stage3_uniform_residual/20260627_stage3_per_image_jsonl_smoke_b4_g4_kodak4/summary.json
per-image:
  results/bitstreams/stage3_uniform_residual/20260627_stage3_per_image_jsonl_smoke_b4_g4_kodak4/per_image_metrics.jsonl

num_records: 4
roundtrip_failure_count: 0
actual_payload_bpp_mean: 0.015015
debug_full_stream_bpp_mean: 0.020020
```

This is the analysis hook for future per-image winner reports across b4 g4,
b4 g8, b5 g4, b5 g8, and any revived hybrid selector.

Added helper:

```text
scripts/compare_stage3_per_image_metrics.py
tests/test_stage3_compare_metrics.py
```

## Bitstream Robustness Smoke Tests

Added tests that reject corrupted, truncated, and trailing-byte compact
CoSERBitstream payloads:

```text
tests/test_bitstream.py
  test_compact_bitstream_crc32_checksum_roundtrip_and_corruption
  test_compact_bitstream_rejects_truncation_and_trailing_bytes

verification:
  pytest tests/test_bitstream.py -q
  7 passed
```

This is not a full adversarial robustness claim. It is the first audit layer for
the security track: malformed transmitted bytes must fail loudly before any
semantic/detail decode is trusted.

## Stage 1 LPIPS Fine-Tune Revisit

Decision note:

```text
docs/research/design_decisions/018_stage1_lpips_finetune_revisit.md
```

Implementation fix:

```text
src/coserdic/losses/stage1.py
  LPIPS branch now runs in fp32 outside CUDA autocast.
  This fixed NaN gradient norm at step 1 for Stage 1 LPIPS fine-tuning.

scripts/eval_stage1_semantic_bitstream.py
  Added eval protocols, perceptual metrics, per-image JSONL, and reconstruction export.
```

Semantic-only CoSER common LIC result:

```text
active5k fixed_bits:
  actual_payload_bpp: 0.012695
  PSNR:               20.918730
  MS-SSIM:             0.689128
  LPIPS:               0.700121
  DISTS:               0.414725

lpips002_ft500 fixed_bits:
  checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_active5k_lpips002_fp32loss_ft500.pt
  actual_payload_bpp: 0.012695
  PSNR:               20.836227
  MS-SSIM:             0.690505
  LPIPS:               0.627592
  DISTS:               0.391976

per-image wins:
  LPIPS: 163 / 165
  DISTS: 152 / 165
  PSNR:   68 / 165
```

Short Stage 2/3 reconnection:

```text
token export:
  outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_lpips002_ft500_fullroots
  global_entropy_bits: 12.259037

token prior:
  checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_3kstep_4096tokens_stage1_lpips002_ft500_lr1e4_do02_es.pt
  best_val_bits_per_token: 11.662687
  final_val_top64_hit: 0.183194

topk512 escape prior:
  outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_4096tokens_stage1_lpips002_ft500_lowlr3k
  fit mean_payload_bpp_image_size: 0.009382
  fit topk_hit_rate: 0.795284

residual prior:
  outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_4096calib_stage1_lpips002_ft500
  mean_huffman_bits_per_symbol: 2.722899
  mean_residual_abs: 0.021944

b4 residual prior:
  outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_4096calib_stage1_lpips002_ft500
  wandb: wandb/offline-run-20260627_182718-7w2wdazq
  mean_huffman_bits_per_symbol: 1.815659
  mean_residual_abs: 0.021944
```

Stage 3 CoSER common LIC comparison:

```text
active b5 g4 unsharp2:
  actual_payload_bpp: 0.018370
  PSNR:               21.344658
  MS-SSIM:             0.693968
  LPIPS:               0.679759
  DISTS:               0.399230

lpips002_ft500 b5 g4 unsharp2:
  run: 20260627_stage3_b5_semposleft_g4_coser_common_lic_stage1_lpips002_ft500_pp_unsharp200_perceptual
  wandb: wandb/offline-run-20260627_181504-medno1p8
  actual_payload_bpp: 0.020490
  semantic_bpp:       0.012453
  detail_bpp:         0.008037
  PSNR:               21.269642
  MS-SSIM:             0.695195
  LPIPS:               0.592988
  DISTS:               0.374476
  semantic_topk_hit:   0.373106
  roundtrip failures: 0

per-image wins over active b5 g4 unsharp2:
  LPIPS: 163 / 165
  DISTS: 155 / 165
  PSNR:   53 / 165
  bpp delta: +0.002120

lpips002_ft500 b4 g4 unsharp2:
  run: 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_lpips002_ft500_pp_unsharp200_perceptual
  wandb: wandb/offline-run-20260627_182913-5ydqtv8a
  actual_payload_bpp: 0.017688
  semantic_bpp:       0.012453
  detail_bpp:         0.005235
  PSNR:               21.217353
  MS-SSIM:             0.694143
  LPIPS:               0.597244
  DISTS:               0.374431
  semantic_topk_hit:   0.373106
  roundtrip failures: 0

per-image wins over active b5 g4 unsharp2:
  LPIPS: 163 / 165
  DISTS: 156 / 165
  PSNR:   38 / 165
  bpp delta: -0.000682

lpips002_ft500 b4 g4 none:
  run: 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_lpips002_ft500_perceptual
  wandb: wandb/offline-run-20260627_183556-rcs4pamx
  actual_payload_bpp: 0.017688
  PSNR:               21.298414
  MS-SSIM:             0.695166
  LPIPS:               0.626042
  DISTS:               0.387196

lpips002_ft500 b4 g4 unsharp0.5:
  run: 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_lpips002_ft500_pp_unsharp050_perceptual
  wandb: wandb/offline-run-20260627_184451-35i0i76b
  actual_payload_bpp: 0.017688
  PSNR:               21.281879
  MS-SSIM:             0.694981
  LPIPS:               0.618435
  DISTS:               0.383264

lpips002_ft500 b4 g4 unsharp1:
  run: 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_lpips002_ft500_pp_unsharp100_perceptual
  wandb: wandb/offline-run-20260627_184232-vcwvwall
  actual_payload_bpp: 0.017688
  PSNR:               21.262786
  MS-SSIM:             0.694748
  LPIPS:               0.611016
  DISTS:               0.379559
```

Reconstruction export + distribution metrics:

```text
run:
  20260627_stage3_b5_semposleft_g4_coser_common_lic_stage1_lpips002_ft500_pp_unsharp200_recon_export
wandb:
  wandb/offline-run-20260627_182250-i4mmiqjl
reconstruction_count:
  165

active b5 g4 unsharp2:
  image FID / KID:    282.247165 / 0.082436
  patch128 FID / KID: 222.928785 / 0.107602

lpips002_ft500 semantic-only:
  image FID / KID:    283.900135 / 0.093163
  patch128 FID / KID: 227.828590 / 0.105361

lpips002_ft500 b4 g4 none:
  image FID / KID:    284.082780 / 0.096637
  patch128 FID / KID: 230.860580 / 0.112192

lpips002_ft500 b4 g4 unsharp1:
  image FID / KID:    285.469126 / 0.101432
  patch128 FID / KID: 231.349338 / 0.114421

lpips002_ft500 b4 g4 unsharp2:
  image FID / KID:    289.087749 / 0.109031
  patch128 FID / KID: 235.915591 / 0.123656

lpips002_ft500 b5 g4 unsharp2:
  image FID / KID:    287.180923 / 0.105697
  patch128 FID / KID: 233.398627 / 0.119745
```

The LPIPS fine-tuned branch is therefore not a clean replacement: it improves
LPIPS/DISTS strongly, but distribution metrics currently regress. The most
likely failure mode is that semantic LPIPS fine-tuning improves local perceptual
features while the current b5 residual/postprocess path shifts global image
statistics. Full 32768-token entropy training is still worthwhile, but residual
gain/range/postprocess must be retuned before promotion.

The new b4 result is stronger for rate-perception than b5: it keeps almost all
of the LPIPS/DISTS gain, reduces actual payload bpp below active b5 unsharp2,
and still roundtrips exactly. It should be treated as a serious perceptual
low-bitrate candidate, while active b4 remains the lower-rate RD anchor. The
postprocess/no-postprocess split shows a useful tradeoff: unsharp2 improves
LPIPS/DISTS, but worsens FID/KID; no postprocess keeps better PSNR/MS-SSIM and
less severe distribution drift. The b4 sweep is monotonic: stronger unsharp
improves LPIPS/DISTS while worsening PSNR/MS-SSIM/FID/KID. Current recommended
reporting split is b4 none for faithfulness-leaning perceptual, b4 unsharp1 for
balanced perception, and b4 unsharp2 for LPIPS/DISTS-first.

Interpretation:

```text
LPIPS Stage 1 fine-tuning is a strong high-perception branch but not a default
low-rate replacement yet. It improves perceptual quality substantially while
raising semantic payload bpp and lowering semantic top-k hit rate on the common
eval. Next full check is 32768-token export + 8k low-LR token prior before
deciding whether the entropy gap is intrinsic or undertrained.
```

## RDVQ-Inspired Stage 1 Rate-Prior Follow-Up

I rechecked Stage 1 against the official-implementation reference policy. The
Stage 1 code was too self-contained to answer the user's concern cleanly, so I
added a minimal RDVQ-style option without importing an external codec:

```text
src/coserdic/models/semantic_vq.py:
  optional VQ assignment probabilities

scripts/train_stage1_semantic_vq.py:
  frozen token-prior soft CE in bits
  --freeze-codebook
  --force-fp32

tests:
  tests/test_semantic_vq.py covers the optional assignment-probability path
```

Validation:

```text
py_compile:
  src/coserdic/models/semantic_vq.py
  scripts/train_stage1_semantic_vq.py

pytest:
  tests/test_semantic_vq.py
  tests/test_token_prior.py
  tests/test_bitstream.py
  tests/test_eval_protocols.py

result:
  17 passed
```

Rejected aggressive probe:

```text
run:
  20260627_stage1_lpips002_rateprior001_tau01_freezecodebook_ft500
wandb:
  wandb/offline-run-20260627_222815-16m1teiz
checkpoint:
  checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior001_tau01_freezecodebook_ft500.pt
reason:
  accidentally used --loss-vq 1.0 instead of the ft500 branch's 0.05.
semantic-only CoSER common:
  actual_payload_bpp: 0.012695
  PSNR / MS-SSIM: 20.029750 / 0.618641
  LPIPS / DISTS:  0.723929 / 0.438140
```

Accepted conservative probe:

```text
run:
  20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500
wandb:
  wandb/offline-run-20260627_223031-lgqjdte7
checkpoint:
  checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500.pt
settings:
  ft500 init, 500 steps, lr=2e-6
  lpips_sem=0.02
  vq=0.05
  codebook_usage=0.005
  rate_prior=0.0005
  tau=0.1
  frozen codebook, FP32

semantic-only CoSER common:
  wandb: wandb/offline-run-20260627_223114-pg2j9vqi
  actual_payload_bpp: 0.012695
  PSNR / MS-SSIM: 20.858840 / 0.691136
  LPIPS / DISTS:  0.615189 / 0.387139

delta vs ft500 semantic-only:
  PSNR:   +0.022613
  MS-SSIM:+0.000632
  LPIPS:  -0.012403
  DISTS:  -0.004836
```

4096-token entropy reconnection:

```text
token export:
  run: 20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_rateprior0005_tau01_vq005_ft500_fullroots
  wandb: wandb/offline-run-20260627_223223-8kd3wkgd
  active_codes: 7877
  global_entropy_bits: 12.245404
  ft500 baseline entropy: 12.259037

32768-token export:
  run: 20260627_stage2_semantic_tokens_32768_oi_div2k_from_stage1_rateprior0005_tau01_vq005_ft500_fullroots
  wandb: wandb/offline-run-20260627_230659-vvdu9yvx
  active_codes: 8126
  global_entropy_bits: 12.268169
  ft500 baseline active_codes: 8125
  ft500 baseline entropy: 12.281880
  delta: -0.013711 bits/token

d256/l4 prior:
  run: 20260627_stage2_token_prior_tf256_l4_3kstep_4096tokens_stage1_rateprior0005_tau01_vq005_ft500_lr1e4_do02_es
  wandb: wandb/offline-run-20260627_223244-6w1w62yi
  best_val_bits/token: 11.624005
  ft500 baseline:      11.662687
```

Top-k sweep, same 4096-token d256/l4 prior:

```text
topk512:
  fit wandb: wandb/offline-run-20260627_223806-c8ibcby4
  calibration payload bpp: 0.009348
  CoSER Stage 3 semantic bpp: 0.012370
  CoSER Stage 3 total bpp:    0.017609

topk1024:
  fit wandb: wandb/offline-run-20260627_225821-9ee54ekx
  calibration payload bpp: 0.009176
  CoSER Stage 3 semantic bpp: 0.012538
  CoSER Stage 3 total bpp:    0.017778

topk2048:
  fit wandb: wandb/offline-run-20260627_225241-eu03mcxp
  calibration payload bpp: 0.009100
  CoSER Stage 3 semantic bpp: 0.012557
  CoSER Stage 3 total bpp:    0.017796
```

Reading: larger top-k improves hit rate but worsens actual payload on CoSER
common for this short prior because rank-event coding gets too expensive.
Use topk512 for this branch until a stronger token prior is trained.

Stage 3 b4 result:

```text
candidate none:
  run: 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_rateprior0005_tau01_vq005_ft500_perceptual
  wandb: wandb/offline-run-20260627_224419-2tlzhfyu
  actual_payload_bpp: 0.017609
  semantic/detail bpp: 0.012370 / 0.005239
  PSNR / MS-SSIM: 21.283043 / 0.695464
  LPIPS / DISTS:  0.614341 / 0.383102
  roundtrip: semantic and detail OK

candidate unsharp2:
  run: 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_rateprior0005_tau01_vq005_ft500_pp_unsharp200_perceptual
  wandb: wandb/offline-run-20260627_224618-e2lv8gz3
  actual_payload_bpp: 0.017609
  PSNR / MS-SSIM: 21.187439 / 0.694023
  LPIPS / DISTS:  0.584765 / 0.372274
  roundtrip: semantic and detail OK

candidate 8192 residual calibration:
  prior run: 20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_stage1_rateprior0005_tau01_vq005_ft500
  prior wandb: wandb/offline-run-20260627_231030-er8nwp6l
  mean_huffman_bits_per_symbol: 1.818209

candidate topk512 b4 8192resid none:
  run: 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_rateprior0005_tau01_vq005_ft500_topk512_8192resid_perceptual
  wandb: wandb/offline-run-20260627_231226-07h66uq0
  actual_payload_bpp: 0.017592
  semantic/detail bpp: 0.012370 / 0.005222
  PSNR / MS-SSIM: 21.283043 / 0.695464
  LPIPS / DISTS:  0.614341 / 0.383102

candidate topk512 b4 8192resid unsharp2:
  run: 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_rateprior0005_tau01_vq005_ft500_topk512_8192resid_pp_unsharp200_perceptual
  wandb: wandb/offline-run-20260627_231426-84wb9jan
  actual_payload_bpp: 0.017592
  PSNR / MS-SSIM: 21.187439 / 0.694023
  LPIPS / DISTS:  0.584765 / 0.372274
```

Per-image wins versus ft500 b4:

```text
none:
  LPIPS: 156 / 165
  DISTS: 130 / 165
  PSNR:   67 / 165
  mean actual_payload_bpp delta: -0.000078

unsharp2:
  LPIPS: 154 / 165
  DISTS: 111 / 165
  PSNR:   55 / 165
  mean actual_payload_bpp delta: -0.000078
```

Decision:

```text
This rate-prior Stage 1 fine-tune is a promising perceptual branch and should
be continued. It beats the baseline ft500 topk512 branch on bpp and
LPIPS/DISTS, but it does not yet beat the stronger ft500 d384/l6 topk2048
entropy setting:

  ft500 d384/l6 topk2048 b4 none:
    actual_payload_bpp: 0.015850
    semantic/detail bpp: 0.010627 / 0.005223

Next promotion run:
  export 32768 candidate tokens, train a d384/l6 token prior, fit topk512 and
  topk2048, then re-run Stage 3 b4 and distribution metrics.
  The 32768 candidate token export is now complete.
```

Candidate recon export and distribution metrics:

```text
run: 20260627_stage3_b4_rateprior0005_topk512_8192resid_unsharp200_coser_common_recon_export
wandb: wandb/offline-run-20260627_232113-4ij10w5p
reconstruction manifest:
  results/bitstreams/stage3_uniform_residual/20260627_stage3_b4_rateprior0005_topk512_8192resid_unsharp200_coser_common_recon_export/reconstructions/manifest.jsonl

actual_payload_bpp: 0.017593
PSNR / MS-SSIM: 21.187342 / 0.694029
LPIPS / DISTS:  0.584753 / 0.372268
roundtrip failures: 0

image FID / KID:             295.411969 / 0.118759
patch128 stride128 FID / KID: 243.197491 / 0.129927
patches: 660
```

Distribution metrics do not promote this candidate over the current ft500
d384/l6 topk2048 family, but they preserve the useful signal that the
rate-aware Stage 1 improves LPIPS/DISTS on most images. The next fair test is
therefore a matching d384/l6 32768-token token-prior run.

Summary artifact:

```text
results/analysis/stage1_lpips_ft/20260627_rateprior0005_tau01_vq005_stage1_stage3_summary.json
```

Next long-run runbook:

```text
docs/research/019_rateprior0005_next_long_runs.md
```
