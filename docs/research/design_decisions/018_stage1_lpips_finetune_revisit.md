# Stage 1 LPIPS Fine-Tune Revisit

Date: 2026-06-27  
Status: Active perceptual branch candidate  
Parent policy: `docs/research/design_decisions/003_official_implementation_reference_policy.md`

## Decision

Do not replace the active Stage 1 checkpoint yet.

Promote the LPIPS fine-tuned Stage 1 checkpoint to a tracked perceptual branch:

```text
checkpoint:
  checkpoints/stage1_semantic_vq/20260627_stage1_active5k_lpips002_fp32loss_ft500.pt

branch role:
  high-perception / rate-perception candidate

not yet:
  default low-rate anchor
```

Rationale:

```text
1. LPIPS fine-tuning gives a large perceptual gain at fixed Stage 1 token count.
2. The Stage 1 architecture and CoSER semantic/detail split are unchanged.
3. The token distribution becomes harder for Stage 2 entropy coding.
4. Full replacement needs a 32k-token export, longer token-prior training, and
   fresh Stage 3 residual-prior calibration.
```

## Implementation Change

Stage 1 LPIPS loss is now computed in fp32 outside CUDA autocast:

```text
file:
  src/coserdic/losses/stage1.py

reason:
  LPIPS backward produced NaN gradients at step 1 when evaluated inside the
  mixed-precision Stage 1 training region.

verification:
  20260627_stage1_active5k_lpips002_fp32loss_ft100 completed without NaNs.
  20260627_stage1_active5k_lpips002_fp32loss_ft500 completed without NaNs.
```

This does not change the default Stage 1 route. `lpips_sem` remains disabled by
default in `configs/train/train_stage1_semantic_vq.yaml`.

## Stage 1 Semantic-Only Result

CoSER common LIC protocol:

```text
dataset:
  Kodak 24 + CLIC Professional Validation 41 + DIV2K Validation 100

codec:
  fixed_bits semantic token stream

actual_payload_bpp:
  0.012695 for both active and LPIPS fine-tuned checkpoints
```

Summary:

```text
active5k:
  PSNR:     20.918730
  MS-SSIM:  0.689128
  LPIPS:    0.700121
  DISTS:    0.414725

lpips002_ft500:
  PSNR:     20.836227
  MS-SSIM:  0.690505
  LPIPS:    0.627592
  DISTS:    0.391976
```

Per-image wins over active5k:

```text
LPIPS lower: 163 / 165
DISTS lower: 152 / 165
PSNR higher:  68 / 165
```

Artifacts:

```text
active:
  results/bitstreams/stage1_semantic_vq/20260627_stage1_active5k_coser_common_lic_perimage_recon/

candidate:
  results/bitstreams/stage1_semantic_vq/20260627_stage1_active5k_lpips002_ft500_coser_common_lic_perimage_recon/

comparison:
  results/analysis/stage1_lpips_ft/20260627_active5k_vs_lpips002_ft500_lpips.json
  results/analysis/stage1_lpips_ft/20260627_active5k_vs_lpips002_ft500_dists.json
  results/analysis/stage1_lpips_ft/20260627_active5k_vs_lpips002_ft500_psnr.json
```

## Short Stage 2 / Stage 3 Reconnection Smoke

Short reconnection used:

```text
token export:
  outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_lpips002_ft500_fullroots/

token prior:
  checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_3kstep_4096tokens_stage1_lpips002_ft500_lr1e4_do02_es.pt

top-k prior:
  outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_4096tokens_stage1_lpips002_ft500_lowlr3k/

residual prior:
  outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_4096calib_stage1_lpips002_ft500/
```

Token entropy observation:

```text
active 4096-token export:
  global_entropy_bits: 11.919707

lpips002_ft500 4096-token export:
  global_entropy_bits: 12.259037

effect:
  LPIPS fine-tuning improves visual semantic reconstruction but increases
  token diversity, making Stage 2 entropy coding harder.
```

Stage 3 common LIC comparison:

```text
active b5 g4 none:
  actual_payload_bpp: 0.018370
  PSNR:               21.390054
  MS-SSIM:             0.693741
  LPIPS:               0.694714
  DISTS:               0.409676

active b5 g4 unsharp2:
  actual_payload_bpp: 0.018370
  PSNR:               21.344658
  MS-SSIM:             0.693968
  LPIPS:               0.679759
  DISTS:               0.399230

lpips002_ft500 b5 g4 none:
  actual_payload_bpp: 0.020490
  semantic_bpp:       0.012453
  detail_bpp:         0.008037
  PSNR:               21.356130
  MS-SSIM:             0.696215
  LPIPS:               0.621523
  DISTS:               0.387300

lpips002_ft500 b5 g4 unsharp2:
  actual_payload_bpp: 0.020490
  semantic_bpp:       0.012453
  detail_bpp:         0.008037
  PSNR:               21.269642
  MS-SSIM:             0.695195
  LPIPS:               0.592988
  DISTS:               0.374476

lpips002_ft500 b4 g4 none:
  actual_payload_bpp: 0.017688
  semantic_bpp:       0.012453
  detail_bpp:         0.005235
  PSNR:               21.298414
  MS-SSIM:             0.695166
  LPIPS:               0.626042
  DISTS:               0.387196

lpips002_ft500 b4 g4 unsharp0.5:
  actual_payload_bpp: 0.017688
  PSNR:               21.281879
  MS-SSIM:             0.694981
  LPIPS:               0.618435
  DISTS:               0.383264

lpips002_ft500 b4 g4 unsharp1:
  actual_payload_bpp: 0.017688
  PSNR:               21.262786
  MS-SSIM:             0.694748
  LPIPS:               0.611016
  DISTS:               0.379559

lpips002_ft500 b4 g4 unsharp2:
  actual_payload_bpp: 0.017688
  semantic_bpp:       0.012453
  detail_bpp:         0.005235
  PSNR:               21.217353
  MS-SSIM:             0.694143
  LPIPS:               0.597244
  DISTS:               0.374431
```

Per-image wins for `lpips002_ft500 b5 g4 unsharp2` over `active b5 g4 unsharp2`:

```text
LPIPS lower: 163 / 165
DISTS lower: 155 / 165
PSNR higher:  53 / 165

rate:
  candidate_minus_reference_actual_payload_bpp: +0.002120
```

Per-image wins for `lpips002_ft500 b4 g4 unsharp2` over `active b5 g4 unsharp2`:

```text
LPIPS lower: 163 / 165
DISTS lower: 156 / 165
PSNR higher:  38 / 165

rate:
  candidate_minus_reference_actual_payload_bpp: -0.000682
```

This is the strongest short-run finding from this revisit: the b4 residual
variant trades away PSNR, but it beats the active b5 unsharp2 branch on
LPIPS/DISTS at lower actual payload bpp.

Distribution metrics on exported reconstructions:

```text
protocol:
  CoSER common LIC, 165 images, 256x256 crop
  Inception-V3 FID/KID
  patch128 uses patch_size=128, patch_stride=128

active b5 g4 unsharp2:
  image FID:      282.247165
  image KID:        0.082436
  patch128 FID:   222.928785
  patch128 KID:     0.107602

lpips002_ft500 semantic-only:
  image FID:      283.900135
  image KID:        0.093163
  patch128 FID:   227.828590
  patch128 KID:     0.105361

lpips002_ft500 b4 g4 none:
  image FID:      284.082780
  image KID:        0.096637
  patch128 FID:   230.860580
  patch128 KID:     0.112192

lpips002_ft500 b4 g4 unsharp1:
  image FID:      285.469126
  image KID:        0.101432
  patch128 FID:   231.349338
  patch128 KID:     0.114421

lpips002_ft500 b4 g4 unsharp2:
  image FID:      289.087749
  image KID:        0.109031
  patch128 FID:   235.915591
  patch128 KID:     0.123656

lpips002_ft500 b5 g4 unsharp2:
  image FID:      287.180923
  image KID:        0.105697
  patch128 FID:   233.398627
  patch128 KID:     0.119745
```

Distribution artifacts:

```text
recon export:
  results/bitstreams/stage3_uniform_residual/20260627_stage3_b5_semposleft_g4_coser_common_lic_stage1_lpips002_ft500_pp_unsharp200_recon_export/

metrics:
  results/analysis/image_distribution_metrics/20260627_stage1_lpips_ft500_semantic_only_coser_common_lic_image_fid_kid.json
  results/analysis/image_distribution_metrics/20260627_stage1_lpips_ft500_semantic_only_coser_common_lic_patch128_s128_fid_kid.json
  results/analysis/image_distribution_metrics/20260627_stage3_lpips_ft500_b4_coser_common_lic_image_fid_kid.json
  results/analysis/image_distribution_metrics/20260627_stage3_lpips_ft500_b4_coser_common_lic_patch128_s128_fid_kid.json
  results/analysis/image_distribution_metrics/20260627_stage3_lpips_ft500_b4_unsharp100_coser_common_lic_image_fid_kid.json
  results/analysis/image_distribution_metrics/20260627_stage3_lpips_ft500_b4_unsharp100_coser_common_lic_patch128_s128_fid_kid.json
  results/analysis/image_distribution_metrics/20260627_stage3_lpips_ft500_b4_unsharp200_coser_common_lic_image_fid_kid.json
  results/analysis/image_distribution_metrics/20260627_stage3_lpips_ft500_b4_unsharp200_coser_common_lic_patch128_s128_fid_kid.json
  results/analysis/image_distribution_metrics/20260627_stage3_lpips_ft500_unsharp200_coser_common_lic_image_fid_kid.json
  results/analysis/image_distribution_metrics/20260627_stage3_lpips_ft500_unsharp200_coser_common_lic_patch128_s128_fid_kid.json
```

## 32k Entropy Reconnection Result

The larger entropy reconnection was completed instead of leaving it as a
recommended future run.

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

Entropy observations:

```text
32768-token export:
  lpips002_ft500 global_entropy_bits: 12.281880
  active5k token-prior val bits:       9.883720
  lpips002_ft500 token-prior val bits: 10.343893

learned topk512 measured on 32768-token calibration:
  topk_hit_rate:                 0.652379
  escape_rate:                   0.347621
  mean_unpadded_bits_per_token: 10.676670
  mean_payload_bpp_image_size:   0.010480

stronger d384/l6 token prior on the same 32768-token export:
  checkpoint:
    checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf384_l6_12kstep_32768tokens_stage1_lpips002_ft500_lr1e4_do02_es.pt
  wandb:
    wandb/offline-run-20260627_192905-99zdasv2
  best_val_bits_per_token:       10.193438
  final_top1 / top5 / top64:      0.045201 / 0.115841 / 0.330190

d384/l6 learned topk512 measured on 32768-token calibration:
  prior:
    outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_stage1_lpips002_ft500_tf384_l6_12k/
  wandb:
    wandb/offline-run-20260627_194728-0e3l9l4y
  topk_hit_rate:                 0.718921
  escape_rate:                   0.281079
  mean_unpadded_bits_per_token: 10.106184
  mean_payload_bpp_image_size:   0.009923

d384/l6 top-k width sweep:
  topk256:
    topk_hit_rate:               0.607009
    escape_rate:                 0.392991
    mean_unpadded_bits_per_token: 10.360242
    mean_payload_bpp_image_size: 0.010171
    wandb: wandb/offline-run-20260627_200429-vir56fm2
  topk1024:
    topk_hit_rate:               0.828103
    escape_rate:                 0.171897
    mean_unpadded_bits_per_token: 9.948924
    mean_payload_bpp_image_size: 0.009769
    wandb: wandb/offline-run-20260627_203000-1picscm6
  topk2048:
    topk_hit_rate:               0.921675
    escape_rate:                 0.078325
    mean_unpadded_bits_per_token: 9.825860
    mean_payload_bpp_image_size: 0.009649
    wandb: wandb/offline-run-20260627_214048-j2ofqg13
  decision:
    topk2048 is the best measured semantic prior for payload bpp. Its
    fit/measurement pass is much slower, so topk1024 remains a faster
    engineering fallback while topk2048 is the paper-bpp candidate.
```

Main Stage 3 variants after 32k semantic prior and 8192 residual prior:

```text
shared:
  run: 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_lpips002_ft500_32kprior_8192resid_pp_unsharp200_perceptual
  wandb: wandb/offline-run-20260627_191835-4a8p89bo
  actual_payload_bpp: 0.016248
  semantic_bpp:       0.011025
  detail_bpp:         0.005223

lpips002_ft500 b4 g4 none 32k/8192:
  run: 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_lpips002_ft500_32kprior_8192resid_perceptual
  wandb: wandb/offline-run-20260627_192703-v8pgpwvu
  PSNR:              21.298414
  MS-SSIM:            0.695166
  LPIPS:              0.626042
  DISTS:              0.387196

lpips002_ft500 b4 g4 unsharp1 32k/8192:
  run: 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_lpips002_ft500_32kprior_8192resid_pp_unsharp100_perceptual
  wandb: wandb/offline-run-20260627_192510-gmx7wmf5
  PSNR:              21.262786
  MS-SSIM:            0.694748
  LPIPS:              0.611016
  DISTS:              0.379559

lpips002_ft500 b4 g4 unsharp2 32k/8192:
  PSNR:              21.217353
  MS-SSIM:            0.694143
  LPIPS:              0.597244
  DISTS:              0.374431
```

Stronger d384/l6 semantic-prior variants at the same checkpoint, residual
quantizer, and postprocess settings:

```text
shared topk512:
  semantic prior:
    outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_stage1_lpips002_ft500_tf384_l6_12k/learned_topk_escape_huffman_prior.json
  token prior:
    checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf384_l6_12kstep_32768tokens_stage1_lpips002_ft500_lr1e4_do02_es.pt
  actual_payload_bpp: 0.016080
  semantic_bpp:       0.010857
  detail_bpp:         0.005223
  semantic_topk_hit:  0.607386
  roundtrip:          semantic=true, detail=true

none:
  run: 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_lpips002_ft500_tf384_32kprior_8192resid_perceptual
  wandb: wandb/offline-run-20260627_195508-g06asqsi
  PSNR / MS-SSIM: 21.298414 / 0.695166
  LPIPS / DISTS: 0.626042 / 0.387196

unsharp1:
  run: 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_lpips002_ft500_tf384_32kprior_8192resid_pp_unsharp100_perceptual
  wandb: wandb/offline-run-20260627_195236-229vpeqt
  PSNR / MS-SSIM: 21.262786 / 0.694748
  LPIPS / DISTS: 0.611016 / 0.379559

unsharp2:
  run: 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_lpips002_ft500_tf384_32kprior_8192resid_pp_unsharp200_perceptual
  wandb: wandb/offline-run-20260627_195007-seuasale
  PSNR / MS-SSIM: 21.217353 / 0.694143
  LPIPS / DISTS: 0.597244 / 0.374431

shared topk1024:
  semantic prior:
    outputs/stage2_learned_entropy/20260627_stage2_learned_topk1024_escape_huffman_fit_32768tokens_stage1_lpips002_ft500_tf384_l6_12k/learned_topk_escape_huffman_prior.json
  actual_payload_bpp: 0.015985
  semantic_bpp:       0.010761
  detail_bpp:         0.005223
  semantic_topk_hit:  0.732102
  roundtrip:          semantic=true, detail=true

topk1024 none:
  run: 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_lpips002_ft500_tf384_topk1024_32kprior_8192resid_perceptual
  wandb: wandb/offline-run-20260627_203802-o0mbxyay
  PSNR / MS-SSIM: 21.298414 / 0.695166
  LPIPS / DISTS: 0.626042 / 0.387196

topk1024 unsharp1:
  run: 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_lpips002_ft500_tf384_topk1024_32kprior_8192resid_pp_unsharp100_perceptual
  wandb: wandb/offline-run-20260627_203526-vhgg7k36
  PSNR / MS-SSIM: 21.262786 / 0.694748
  LPIPS / DISTS: 0.611016 / 0.379559

topk1024 unsharp2:
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
  roundtrip:          semantic=true, detail=true
  PSNR / MS-SSIM:     21.298414 / 0.695166
  LPIPS / DISTS:      0.626042 / 0.387196

topk2048 unsharp1:
  run: 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_lpips002_ft500_tf384_topk2048_32kprior_8192resid_pp_unsharp100_perceptual
  wandb: wandb/offline-run-20260627_220229-7003wutq
  actual_payload_bpp: 0.015850
  semantic_bpp:       0.010627
  detail_bpp:         0.005223
  semantic_topk_hit:  0.855398
  roundtrip:          semantic=true, detail=true
  PSNR / MS-SSIM:     21.262786 / 0.694748
  LPIPS / DISTS:      0.611016 / 0.379559

topk2048 unsharp2:
  run: 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_lpips002_ft500_tf384_topk2048_32kprior_8192resid_pp_unsharp200_perceptual
  wandb: wandb/offline-run-20260627_214350-wk0otkqf
  actual_payload_bpp: 0.015850
  semantic_bpp:       0.010627
  detail_bpp:         0.005223
  semantic_topk_hit:  0.855398
  roundtrip:          semantic=true, detail=true
  PSNR / MS-SSIM:     21.217353 / 0.694143
  LPIPS / DISTS:      0.597244 / 0.374431
```

The reconstruction metrics are unchanged relative to the earlier b4/unsharp2
branch because the checkpoint, residual quantizer, and fixed postprocess are
unchanged. The improvement is in the actual transmitted payload:

```text
vs lpips002_ft500 b4 g4 unsharp2 4k prior:
  actual_payload_bpp: 0.017688 -> 0.016248
  bpp delta:          -0.001440
  quality delta:       unchanged

vs lpips002_ft500 b4 g4 unsharp2 d256 32k/8192 prior:
  actual_payload_bpp: 0.016248 -> 0.016080
  semantic_bpp:       0.011025 -> 0.010857
  bpp delta:          -0.000168
  quality delta:       unchanged

vs lpips002_ft500 b4 g4 unsharp2 d384 topk512 prior:
  actual_payload_bpp: 0.016080 -> 0.015985
  semantic_bpp:       0.010857 -> 0.010761
  bpp delta:          -0.000095
  quality delta:       unchanged

vs lpips002_ft500 b4 g4 unsharp2 d384 topk1024 prior:
  actual_payload_bpp: 0.015985 -> 0.015850
  semantic_bpp:       0.010761 -> 0.010627
  bpp delta:          -0.000135
  quality delta:       unchanged

vs active b5 g4 unsharp2:
  actual_payload_bpp: 0.018370 -> 0.015850
  bpp delta:          -0.002520
  LPIPS delta:        -0.082515
  DISTS delta:        -0.024799
  PSNR delta:         -0.127305
```

GenCodec reproduction protocol check, strict 552 images:

```text
datasets:
  Kodak 24
  CLIC2020 test 428 = professional/test 250 + mobile/test 178
  DIV2K val 100 = 0801.png-0900.png from /dpl/div2k

topk1024 unsharp2:
  run: 20260627_stage3_b4_semposleft_g4_gencodec_reproduction_stage1_lpips002_ft500_tf384_topk1024_32kprior_8192resid_pp_unsharp200_perceptual
  wandb: wandb/offline-run-20260627_205054-fjbibejz
  actual_payload_bpp: 0.015409
  semantic_bpp:       0.010236
  detail_bpp:         0.005173
  semantic_topk_hit:  0.769248
  PSNR / MS-SSIM:     21.931414 / 0.723483
  LPIPS / DISTS:      0.562911 / 0.364492
  roundtrip:          semantic=true, detail=true

topk2048 unsharp2:
  run: 20260627_stage3_b4_semposleft_g4_gencodec_reproduction_stage1_lpips002_ft500_tf384_topk2048_32kprior_8192resid_pp_unsharp200_perceptual
  wandb: wandb/offline-run-20260627_215239-g5mfih6n
  actual_payload_bpp: 0.015293
  semantic_bpp:       0.010120
  detail_bpp:         0.005173
  semantic_topk_hit:  0.877151
  PSNR / MS-SSIM:     21.931414 / 0.723483
  LPIPS / DISTS:      0.562911 / 0.364492
  roundtrip:          semantic=true, detail=true
```

Per-image wins over `active b5 g4 unsharp2`:

```text
lower actual_payload_bpp: 161 / 165
LPIPS lower:             163 / 165
DISTS lower:             156 / 165
PSNR higher:              38 / 165
```

Distribution metrics for `lpips002_ft500 b4 g4 unsharp2 32k/8192`:

```text
image FID:      289.087749
image KID:        0.107389
patch128 FID:   235.915591
patch128 KID:     0.122006

artifacts:
  results/analysis/stage1_lpips_ft/20260627_stage3_b4_ft500_32kprior_8192resid_unsharp200_image_distribution.json
  results/analysis/stage1_lpips_ft/20260627_stage3_b4_ft500_32kprior_8192resid_unsharp200_patch128s128_distribution.json
```

Topk2048 fixed-decoder detail-gain distribution probe:

```text
shared:
  actual_payload_bpp: 0.015850
  semantic_bpp:       0.010627
  detail_bpp:         0.005223
  transmitted payload: unchanged across gain values

gain0.5 none:
  run: 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_lpips002_ft500_tf384_topk2048_32kprior_8192resid_gain050_perceptual
  recon export: 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_lpips002_ft500_tf384_topk2048_32kprior_8192resid_gain050_recon_export
  wandb: wandb/offline-run-20260627_220803-2o77kk4f
  PSNR / MS-SSIM: 21.179101 / 0.694154
  LPIPS / DISTS: 0.623141 / 0.389397
  image FID / KID:    282.674784 / 0.090455
  patch128 FID / KID: 227.783365 / 0.104939

gain0.75 none:
  run: 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_lpips002_ft500_tf384_topk2048_32kprior_8192resid_gain075_perceptual
  recon export: 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_lpips002_ft500_tf384_topk2048_32kprior_8192resid_gain075_recon_export
  wandb: wandb/offline-run-20260627_221055-sa6tkorz
  PSNR / MS-SSIM: 21.273588 / 0.694981
  LPIPS / DISTS: 0.623817 / 0.388134
  image FID / KID:    282.912345 / 0.091588
  patch128 FID / KID: 228.972532 / 0.107392

gain1.0 none, previous distribution baseline:
  PSNR / MS-SSIM: 21.298414 / 0.695166
  LPIPS / DISTS: 0.626042 / 0.387196
  image FID / KID:    284.082780 / 0.096637
  patch128 FID / KID: 230.860580 / 0.112192
```

This is a useful fixed-decoder tradeoff. Lowering `detail_gain` does not change
the transmitted payload because the residual codes are still sent; it only
changes the deterministic decoder reconstruction. `gain0.5 none` is currently
the safest FID/KID point for the ft500 branch and nearly closes the image-FID
gap to the active b5 unsharp2 anchor, but it gives up most of the LPIPS/DISTS
gain. `gain1.0 unsharp2` remains the LPIPS/DISTS-first point.

## Interpretation

This branch is a strong rate-perception candidate, not a free win:

```text
what improved:
  semantic-only LPIPS/DISTS improved substantially.
  Stage 3 LPIPS/DISTS improved substantially after reconnecting priors.
  32k entropy reconnection recovered most of the ft500 semantic payload gap.
  b4 residual keeps most of the b5 perceptual gain while lowering total bpp
  below active b5 unsharp2 by about 0.00239 actual payload bpp.
  roundtrip remained exact for semantic and detail streams.

what worsened:
  semantic payload bpp remains above the active branch.
  semantic top-k hit rate dropped on CoSER common eval:
    active:                  0.650947
    lpips002_ft500 4k prior: 0.373106
    lpips002_ft500 32k prior: 0.583049
    lpips002_ft500 d384/l6:  0.607386
    lpips002_ft500 topk1024: 0.732102
    lpips002_ft500 topk2048: 0.855398
  b5 detail payload bpp also increased slightly.
  b4 regresses PSNR against active b4/b5 anchors.
  image-level and patch128 FID/KID did not improve over active unsharp2.
  Stage 3 residual improves LPIPS/DISTS but currently worsens ft500 FID/KID
  relative to ft500 semantic-only.
  unsharp2 improves LPIPS but worsens FID/KID relative to no postprocess.
```

Postprocess sweep reading:

```text
b4 none -> unsharp0.5 -> unsharp1 -> unsharp2:
  LPIPS/DISTS improve monotonically.
  PSNR/MS-SSIM decrease monotonically.
  FID/KID worsen from none to unsharp1 to unsharp2.

preferred reporting:
  b4 gain0.5 none:
    FID/KID-safe fixed-decoder candidate
  b4 none:
    RD/faithfulness-leaning perceptual candidate
  b4 unsharp1:
    balanced perceptual candidate
  b4 unsharp2:
    LPIPS/DISTS-first candidate
```

Main hypothesis:

```text
LPIPS fine-tuning pushes the semantic tokenizer toward more visually distinct
code usage. This improves perceptual reconstruction but increases token
entropy. The 32k entropy reconnection shows that much of the short-run bpp
penalty was an entropy-model calibration issue, not an unavoidable Stage 1
cost. The gap does not fully close, but the b4 branch now clearly beats the
active b5/unsharp2 branch on rate-perception.

The distribution-metric result adds a second constraint: the detail residual
path must be retuned for perceptual/global-distribution quality, not just PSNR
or LPIPS. A simple residual branch is still preferred, but the quantization
range/gain/postprocess should be searched before promoting the branch.

The b4 result changes the branch status from "interesting but higher-rate" to
"promising rate-perception candidate". It should still not replace the default
RD anchor, because PSNR and FID/KID need follow-up, but it is a serious
paper-table candidate for perceptual low-bitrate comparison.

## RDVQ-Inspired Stage 1 Rate-Prior Revisit

I rechecked the Stage 1 implementation against the official-implementation
reference policy. RDVQ's useful lesson for CoSER-DiC is not to import their
codec, but to preserve our semantic/detail Core MVP while letting a frozen
token prior gently shape Stage 1 tokens during fine-tuning. The CoSER
implementation therefore remains native, but now supports an optional
rate-prior loss:

```text
implementation:
  src/coserdic/models/semantic_vq.py
    optional differentiable assignment probabilities from VQ

  scripts/train_stage1_semantic_vq.py
    --loss-rate-prior
    --rate-prior-checkpoint
    --rate-soft-temperature
    --freeze-codebook
    --force-fp32

  tests/test_semantic_vq.py
    verifies assignment-probability shape and gradient flow
```

The first aggressive probe is rejected:

```text
run:
  20260627_stage1_lpips002_rateprior001_tau01_freezecodebook_ft500
wandb:
  wandb/offline-run-20260627_222815-16m1teiz
reason:
  inherited --loss-vq 1.0, much stronger than the ft500 branch setting.
semantic eval:
  actual_payload_bpp: 0.012695
  PSNR / MS-SSIM:     20.029750 / 0.618641
  LPIPS / DISTS:      0.723929 / 0.438140
```

The corrected conservative probe is a real candidate:

```text
run:
  20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500
wandb:
  wandb/offline-run-20260627_223031-lgqjdte7
checkpoint:
  checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500.pt
settings:
  init: ft500 LPIPS branch
  steps: 500
  lr: 2e-6
  lpips_sem: 0.02
  vq: 0.05
  codebook_usage: 0.005
  rate_prior: 0.0005
  tau: 0.1
  frozen codebook, FP32
```

Semantic-only CoSER common LIC comparison:

```text
ft500 semantic-only:
  actual_payload_bpp: 0.012695
  PSNR / MS-SSIM:     20.836227 / 0.690505
  LPIPS / DISTS:      0.627592 / 0.391976

rateprior0005 semantic-only:
  actual_payload_bpp: 0.012695
  PSNR / MS-SSIM:     20.858840 / 0.691136
  LPIPS / DISTS:      0.615189 / 0.387139

candidate minus ft500:
  PSNR:   +0.022613
  MS-SSIM:+0.000632
  LPIPS:  -0.012403
  DISTS:  -0.004836
```

Short 4096-token entropy reconnection:

```text
token export:
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

d256/l4 token prior:
  best_val_bits/token: 11.624005
  ft500 baseline:      11.662687

top-k actual Stage 3 reading:
  topk512 is best on CoSER common.
  topk1024/topk2048 improve hit rate but worsen actual payload bpp because
  rank-event coding gets more expensive.
```

Stage 3 b4 CoSER common LIC, 4096-token d256/l4 prior:

```text
ft500 topk512 b4 none:
  actual_payload_bpp: 0.017688
  semantic/detail bpp: 0.012453 / 0.005235
  PSNR / MS-SSIM: 21.298414 / 0.695166
  LPIPS / DISTS:  0.626042 / 0.387196

rateprior0005 topk512 b4 none:
  actual_payload_bpp: 0.017609
  semantic/detail bpp: 0.012370 / 0.005239
  PSNR / MS-SSIM: 21.283043 / 0.695464
  LPIPS / DISTS:  0.614341 / 0.383102
  roundtrip: semantic and detail OK

rateprior0005 topk512 b4 unsharp2:
  actual_payload_bpp: 0.017609
  PSNR / MS-SSIM: 21.187439 / 0.694023
  LPIPS / DISTS:  0.584765 / 0.372274
  roundtrip: semantic and detail OK

rateprior0005 topk512 b4, 8192 residual calibration:
  actual_payload_bpp: 0.017592
  semantic/detail bpp: 0.012370 / 0.005222
  none LPIPS / DISTS:     0.614341 / 0.383102
  unsharp2 LPIPS / DISTS: 0.584765 / 0.372274
```

Per-image wins versus ft500 b4:

```text
none:
  LPIPS: 156 / 165
  DISTS: 130 / 165
  PSNR:   67 / 165
  mean bpp delta: -0.000078

unsharp2:
  LPIPS: 154 / 165
  DISTS: 111 / 165
  PSNR:   55 / 165
  mean bpp delta: -0.000078
```

Decision:

```text
Keep rateprior0005 as a promising perception-oriented Stage 1 candidate.
Do not promote it over the current best ft500 d384/l6 topk2048 branch yet,
because that stronger entropy setting is still much lower rate:

  ft500 d384/l6 topk2048 b4 none:
    actual_payload_bpp: 0.015850
    semantic/detail bpp: 0.010627 / 0.005223

Next promotion test:
  export 32768 candidate tokens, train d384/l6 token prior, fit topk512 and
  topk2048 priors, then re-evaluate Stage 3 b4 and distribution metrics.
  The 32768 candidate token export is already available.
```

Distribution metrics for the candidate recon export:

```text
run:
  20260627_stage3_b4_rateprior0005_topk512_8192resid_unsharp200_coser_common_recon_export
wandb:
  wandb/offline-run-20260627_232113-4ij10w5p
actual_payload_bpp: 0.017593
LPIPS / DISTS:      0.584753 / 0.372268
roundtrip failures: 0

image-level FID / KID:
  295.411969 / 0.118759

patch128 stride128 FID / KID:
  243.197491 / 0.129927
  patches: 660
```

This does not improve the best ft500 topk2048 distribution-metric branch, so
the candidate remains a Stage 1/promising-perceptual branch rather than the
active paper-bpp branch. The important next test is the d384/l6 32768-token
semantic prior, not further reading from the short d256/l4 topk512 prior.

Full machine-readable summary:

```text
results/analysis/stage1_lpips_ft/20260627_rateprior0005_tau01_vq005_stage1_stage3_summary.json
```

Next long-run runbook:

```text
docs/research/019_rateprior0005_next_long_runs.md
```
```

## Next Full Run

The previous 32k reconnection command sequence has now been executed. The next
longer run should focus on improving the ft500 branch without losing its simple
Core MVP structure:

```text
1. Keep d384/l6 topk2048 as the current paper-bpp semantic-prior setting for
   this branch. Use topk1024 as the faster engineering fallback when running
   broad sweeps, because topk2048 fitting/measurement is substantially slower.
2. Try a longer/stronger token prior for ft500 tokens:
   d_model=512 or more calibration tokens if the top-k sweep still leaves a
   clear semantic payload gap.
3. Try a modest Stage 1 regularization run that preserves LPIPS gain while
   reducing token entropy:
   lpips_sem=0.001-0.002 plus entropy/usage penalty tuning.
4. Search residual/postprocess settings for better FID/KID:
   b4 none/unsharp1 as safer reporting points, and avoid relying only on
   unsharp2.
5. Re-run GenCodec reproduction protocol after any checkpoint/residual change;
   current topk2048 validation is already available for the fixed ft500 branch.
```
