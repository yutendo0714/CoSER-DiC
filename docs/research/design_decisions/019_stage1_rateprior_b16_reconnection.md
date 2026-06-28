# Stage 1 Rateprior Batch-16 Reconnection

Date: 2026-06-28 JST  
Status: Active rate-perception candidate  
Parent: `docs/research/design_decisions/018_stage1_lpips_finetune_revisit.md`

## Decision

Keep the batch-16 rate-prior Stage 1 branch as the strongest current
perception-oriented Core MVP candidate, but do not freeze it as the final paper
branch until the top-k2048 semantic prior and distribution metrics are complete.

```text
checkpoint:
  checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt

role:
  Stage 1 semantic tokenizer candidate for low-bitrate perceptual comparison

keep unchanged:
  Core MVP split:
    semantic VQ token stream
    learned top-k escape Huffman semantic entropy
    static semantic-conditioned residual grid entropy
```

The branch keeps the same architecture and does not import an external codec.
The change is a conservative training-protocol refinement: same LPIPS/rate-prior
objective family, larger batch, and then fresh Stage 2/3 calibration.

## Stage 1 Batch Sweep

CoSER common LIC protocol, semantic-only fixed 13-bit token stream:

```text
baseline rateprior0005 b4:
  actual_payload_bpp: 0.012695
  PSNR / MS-SSIM:     20.858840 / 0.691136
  LPIPS / DISTS:      0.615189 / 0.387139

b8:
  checkpoint:
    checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b8.pt
  wandb:
    wandb/offline-run-20260627_233747-1n7k6ct5
  actual_payload_bpp: 0.012695
  PSNR / MS-SSIM:     20.859303 / 0.690701
  LPIPS / DISTS:      0.612573 / 0.386312

b16:
  checkpoint:
    checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt
  wandb:
    wandb/offline-run-20260627_233518-5a72ua3f
  actual_payload_bpp: 0.012695
  PSNR / MS-SSIM:     20.864920 / 0.690404
  LPIPS / DISTS:      0.607793 / 0.384561
```

Batch 16 is the best semantic-only perceptual point in this sweep. Batch 8
reduces the training prior loss more aggressively but does not beat batch 16 on
the evaluation metrics that matter.

## 4096-Token Entropy Warning

The 4096-token exports were useful as quick checks but were not reliable enough
for promotion:

```text
b16 4096 export:
  active_codes: 7881
  global_entropy_bits: 12.250110

b16 d256/l4 prior:
  best_val_bits_per_token: 11.596631

b16 topk512 on calibration:
  mean_payload_bpp_image_size: 0.008973

b16 topk512 on CoSER common:
  actual_payload_bpp: 0.012846
  fixed_bits_payload_bpp: 0.012695
  semantic_topk_hit: 0.392140

b16 topk2048 on CoSER common:
  actual_payload_bpp: 0.012831
  fixed_bits_payload_bpp: 0.012695
  semantic_topk_hit: 0.676610
```

The calibration payload looked strong, but evaluation hit rate collapsed enough
that learned entropy coding became worse than fixed 13-bit coding. Do not use
4096-token top-k results to judge the branch.

## 32768-Token Reconnection

Artifacts:

```text
token export:
  outputs/stage2_semantic_tokens/20260628_stage2_semantic_tokens_32768_oi_div2k_from_stage1_rateprior0005_b16_fullroots/semantic_tokens.pt
  wandb/offline-run-20260628_000203-a2ezynid
  active_codes: 8131
  global_entropy_bits: 12.272262

token prior:
  checkpoints/stage2_token_prior/20260628_stage2_token_prior_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_es.pt
  wandb/offline-run-20260628_000224-lxd8yy1p
  best_val_bits_per_token: 10.153697
  final_top1 / top5 / top64: 0.046451 / 0.118391 / 0.334843

semantic topk512 prior:
  outputs/stage2_learned_entropy/20260628_stage2_learned_topk512_escape_huffman_fit_32768tokens_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json
  wandb/offline-run-20260628_002104-dntzt08o
  calibration topk_hit_rate: 0.723387
  calibration mean_payload_bpp_image_size: 0.009878
```

CoSER common semantic-only learned stream:

```text
run:
  20260628_stage2_b16_topk512_32kprior_tf384_coser_common_perceptual_eval
wandb:
  wandb/offline-run-20260628_002440-bbyjwbrv

learned actual_payload_bpp: 0.010828
fixed actual_payload_bpp:   0.012695
delta vs fixed:            -0.001867
semantic_topk_hit:          0.609280
LPIPS / DISTS:              0.607793 / 0.384561
roundtrip:                  true
```

This is the first reliable signal that the b16 branch can keep its semantic
perceptual gain while recovering real transmitted semantic payload.

## Stage 3 Reconnection

Fresh residual prior for this exact Stage 1 checkpoint:

```text
run:
  20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_stage1_rateprior0005_b16
prior:
  outputs/stage3_residual_entropy/20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json
wandb:
  wandb/offline-run-20260628_002706-c20i20t9

coding:
  semantic_position_leftctx_huffman
  group_count: 4
  smoothing_count: 0
  detail: d32 b4 range0.25

mean_huffman_bits_per_symbol: 1.813093
fixed_bits_per_symbol:        4
mean_residual_abs:            0.021790
residual_std:                 0.030657
clip_ratio:                   0.000140
```

CoSER common Stage 3, topk512 semantic prior:

```text
no postprocess:
  run:
    20260628_stage3_b16_topk512_tf384_semposleft_g4_b4_coser_common_no_pp_perceptual
  wandb:
    wandb/offline-run-20260628_002938-q75d27u9
  actual_payload_bpp: 0.016038
  semantic/detail bpp: 0.010828 / 0.005210
  PSNR / MS-SSIM:     21.288482 / 0.694823
  LPIPS / DISTS:      0.606929 / 0.380292
  roundtrip:          semantic=true, detail=true

unsharp3x3 strength2:
  run:
    20260628_stage3_b16_topk512_tf384_semposleft_g4_b4_coser_common_pp_unsharp200_perceptual
  wandb:
    wandb/offline-run-20260628_003210-moq55w0v
  actual_payload_bpp: 0.016038
  semantic/detail bpp: 0.010828 / 0.005210
  PSNR / MS-SSIM:     21.192536 / 0.693296
  LPIPS / DISTS:      0.577572 / 0.371905
  roundtrip:          semantic=true, detail=true
```

Comparison against the previous ft500 topk2048 b4 branch:

```text
ft500 topk2048 b4 none:
  actual_payload_bpp: 0.015850
  LPIPS / DISTS:      0.626042 / 0.387196

b16 topk512 b4 none:
  actual_payload_bpp: 0.016038
  LPIPS / DISTS:      0.606929 / 0.380292

ft500 topk2048 b4 unsharp2:
  actual_payload_bpp: 0.015850
  LPIPS / DISTS:      0.597244 / 0.374431

b16 topk512 b4 unsharp2:
  actual_payload_bpp: 0.016038
  LPIPS / DISTS:      0.577572 / 0.371905
```

Interpretation:

```text
rate:
  b16 topk512 is currently +0.000188 actual_payload_bpp above ft500 topk2048.

perception:
  b16 is substantially better on LPIPS and slightly better on DISTS.

next rate check:
  topk2048 fitting for b16 completed and reverses the remaining bpp gap.
```

## Top-k2048 Promotion Result

Fitting and evaluation artifacts:

```text
semantic topk2048 prior:
  outputs/stage2_learned_entropy/20260628_stage2_learned_topk2048_escape_huffman_fit_32768tokens_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json
  wandb/offline-run-20260628_012004-kgalcosw

calibration:
  topk_hit_rate:                 0.923410
  escape_rate:                   0.076590
  mean_unpadded_bits_per_token:  9.782553
  mean_payload_bpp_image_size:   0.009607

CoSER common Stage 2 eval:
  run:
    20260628_stage2_b16_topk2048_32kprior_tf384_coser_common_perceptual_eval
  wandb:
    wandb/offline-run-20260628_012205-opfsc5xg
  actual_payload_bpp: 0.010585
  fixed_bits_bpp:     0.012695
  delta vs fixed:    -0.002111
  semantic_topk_hit:  0.857576
  roundtrip:          true
```

Stage 3 with the same b16 residual prior:

```text
topk2048 none:
  run:
    20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_coser_common_no_pp_perceptual
  wandb:
    wandb/offline-run-20260628_012457-gfh73ptp
  actual_payload_bpp: 0.015794
  semantic/detail bpp: 0.010585 / 0.005210
  semantic_topk_hit:  0.857576
  PSNR / MS-SSIM:     21.288482 / 0.694823
  LPIPS / DISTS:      0.606929 / 0.380292
  roundtrip:          semantic=true, detail=true

topk2048 unsharp2:
  run:
    20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_coser_common_pp_unsharp200_perceptual
  wandb:
    wandb/offline-run-20260628_012749-x78oq6di
  actual_payload_bpp: 0.015794
  semantic/detail bpp: 0.010585 / 0.005210
  semantic_topk_hit:  0.857576
  PSNR / MS-SSIM:     21.192536 / 0.693296
  LPIPS / DISTS:      0.577572 / 0.371905
  roundtrip:          semantic=true, detail=true
```

Comparison against the previous ft500 d384/l6 topk2048 branch:

```text
ft500 topk2048 b4 none:
  actual_payload_bpp: 0.015850
  LPIPS / DISTS:      0.626042 / 0.387196

b16 topk2048 b4 none:
  actual_payload_bpp: 0.015794
  LPIPS / DISTS:      0.606929 / 0.380292

ft500 topk2048 b4 unsharp2:
  actual_payload_bpp: 0.015850
  LPIPS / DISTS:      0.597244 / 0.374431

b16 topk2048 b4 unsharp2:
  actual_payload_bpp: 0.015794
  LPIPS / DISTS:      0.577572 / 0.371905
```

This promotes b16/topk2048 to the current internal Core MVP
rate-perception leader on CoSER common LIC. It is not yet a final paper claim
against all external published methods, because GenCodec reproduction protocol,
FID/KID, visual audits, and stronger baselines still need to be refreshed.

## Distribution Check

Exported reconstructions:

```text
unsharp2 export:
  results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_coser_common_pp_unsharp200_recon_export/
  wandb/offline-run-20260628_013200-o8aiwvaw

no-postprocess export:
  results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_coser_common_no_pp_recon_export/
  wandb/offline-run-20260628_013543-65fc161w
```

Distribution metrics:

```text
b16 topk2048 unsharp2:
  image FID / KID:     303.791554 / 0.130428
  patch128_s128 diagnostic FID / KID:  247.876866 / 0.134553

b16 topk2048 none:
  image FID / KID:     286.907047 / 0.100505
  patch128_s128 diagnostic FID / KID:  236.244679 / 0.116280
```

Note: these CoSER common LIC patch128 values are internal torch-fidelity
single-grid diagnostics, not CoD/CoD-Lite dataset-level patch FID.

Artifacts:

```text
results/analysis/image_distribution_metrics/20260628_stage3_b16_topk2048_unsharp200_coser_common_image_fid_kid.json
results/analysis/image_distribution_metrics/20260628_stage3_b16_topk2048_unsharp200_coser_common_patch128_s128_fid_kid.json
results/analysis/image_distribution_metrics/20260628_stage3_b16_topk2048_no_pp_coser_common_image_fid_kid.json
results/analysis/image_distribution_metrics/20260628_stage3_b16_topk2048_no_pp_coser_common_patch128_s128_fid_kid.json
```

Interpretation:

```text
LPIPS/DISTS:
  unsharp2 is the strongest current point.

FID/KID:
  no-postprocess is substantially safer than unsharp2.
  b16 no-postprocess is still slightly weaker than the previous ft500
  no-postprocess distribution metrics, so the branch is not yet a universal
  perceptual/distribution win.

reporting:
  use b16 topk2048 unsharp2 as LPIPS/DISTS-first point.
  use b16 topk2048 none as FID/KID-safer neutral point.
  do not hide the FID/KID regression when discussing this branch.
```

## Per-Image Check

Per-image summaries for b16 topk512 b4 unsharp2:

```text
LPIPS delta vs semantic-only:
  mean: -0.030206
  std:   0.019126
  min:  -0.075022
  max:   0.049527

DISTS delta vs semantic-only:
  mean: -0.012658
  std:   0.012629
  min:  -0.040556
  max:   0.024523

actual_payload_bpp:
  mean: 0.016038
  std:  0.002209
  min:  0.007690
  max:  0.023193
```

Dataset breakdown:

```text
LPIPS delta mean:
  Kodak:     -0.034434
  CLIC Pro:  -0.022668
  DIV2K val: -0.032282

DISTS delta mean:
  Kodak:     -0.012334
  CLIC Pro:  -0.012773
  DIV2K val: -0.012688
```

Worst positive LPIPS deltas:

```text
+0.049527 / DISTS +0.013517:
  /dpl/clic/professional/valid/sergey-zolkin-21232.png

+0.031298 / DISTS -0.016056:
  /dpl/div2k/0893.png

+0.020952 / DISTS -0.014660:
  /dpl/div2k/0866.png
```

Most LPIPS regressions are not DISTS regressions. The main true postprocess
risk case is `sergey-zolkin-21232.png`; future postprocess sweeps should check
strength 1.0 and 1.5, not only strength 2.0.

## GenCodec Protocol Crop-Size Audit

A first 552-image GenCodec-split evaluation was completed with the correct
dataset composition but the wrong crop-size for a CoD/CoD-Lite reproduction
table:

```text
run:
  20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_gencodec_reproduction_pp_unsharp200_perceptual
wandb:
  wandb/offline-run-20260628_014604-fouqqc6s
summary:
  results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_gencodec_reproduction_pp_unsharp200_perceptual/summary.json

dataset composition:
  Kodak 24
  CLIC2020 test 428
  DIV2K validation 100
  count_status: all ok

crop_size:
  actual: 256
  protocol default: 512

actual_payload_bpp: 0.015260
semantic/detail bpp: 0.010076 / 0.005184
PSNR / MS-SSIM: 21.904084 / 0.722570
LPIPS / DISTS: 0.546500 / 0.362395
roundtrip: semantic=true, detail=true
```

Interpretation:

```text
usable as:
  broad dataset-composition diagnostic for the current 256-token-grid Core MVP.

not usable as:
  a strict CoD / CoD-Lite reproduction table entry, because those
  settings use 512 crops or full-resolution variants depending on the paper.
```

A 512-crop shape probe failed as expected:

```text
run:
  20260628_stage3_b16_topk2048_gencodec_crop512_shape_probe
failure:
  ValueError: input length exceeds configured context_length
cause:
  current semantic token prior was trained for 8x8 = 64 tokens.
  512 crop produces 16x16 = 256 tokens.
additional blocker:
  current residual Huffman prior is detail_shape [3, 8, 8], while d32 on
  512 crop needs [3, 16, 16].
```

Code guard added:

```text
scripts/eval_stage1_semantic_bitstream.py
scripts/eval_stage2_static_huffman_bitstream.py
scripts/eval_stage2_learned_topk_escape_bitstream.py
scripts/eval_stage3_uniform_residual_bitstream.py
```

The scripts now use a protocol's default crop-size when `--crop-size` is
omitted, and write `protocol_default_crop_size` plus
`crop_size_matches_protocol_default` to the summary. Future 256-crop
GenCodec-split diagnostics must therefore pass `--crop-size 256` explicitly.

## 512-Crop Rebuild

The strict GenCodec/CoD-style 512-crop path was rebuilt instead of forcing the
64-token prior beyond its context length.

Semantic token export:

```text
run:
  20260628_stage2_semantic_tokens_32768_oi_val_crop512_stage1_rateprior0005_b16
artifact:
  outputs/stage2_semantic_tokens/20260628_stage2_semantic_tokens_32768_oi_val_crop512_stage1_rateprior0005_b16/semantic_tokens.pt

data:
  OpenImages validation crops
  max_images: 32768
  crop_size: 512
  token_shape: [16, 16]

active_codes: 8177
global_entropy_bits: 12.207300
```

Token prior:

```text
run:
  20260628_stage2_token_prior_crop512_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_amp_es
checkpoint:
  checkpoints/stage2_token_prior/20260628_stage2_token_prior_crop512_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_amp_es.pt

context_length: 256
best_step: 11500
best_val_bits_per_token: 8.799404
final_val_bits_per_token: 8.799048
final_top1 / top5 / top64: 0.088480 / 0.201950 / 0.476710
```

Semantic learned top-k escape prior:

```text
run:
  20260628_stage2_learned_topk512_escape_huffman_fit_32768tokens_crop512_stage1_rateprior0005_b16_tf384_l6_12k
prior:
  outputs/stage2_learned_entropy/20260628_stage2_learned_topk512_escape_huffman_fit_32768tokens_crop512_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json

topk: 512
calibration topk_hit_rate: 0.795370
calibration escape_rate: 0.204630
calibration mean_unpadded_bits_per_token: 9.214170
calibration mean_payload_bpp_image_size: 0.009012
fixed_bits_payload_bpp_image_size: 0.012695
```

Residual prior:

```text
run:
  20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_crop512_8192calib_stage1_rateprior0005_b16
prior:
  outputs/stage3_residual_entropy/20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_crop512_8192calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json

detail_shape: [3, 16, 16]
detail_bits: 4
detail_range: 0.25
coding_mode: semantic_position_leftctx_huffman
semantic_group_count: 4
smoothing_count: 0

mean_huffman_bits_per_symbol: 1.725830
fixed_bits_per_symbol: 4
mean_residual_abs: 0.020160
residual_std: 0.028820
clip_ratio: 0.000102
```

Implementation note:

```text
src/coserdic/entropy/topk_escape.py
tests/test_topk_escape.py
```

Top-k event counting and bit-count diagnostics are now tensorized, which makes
top-k escape fitting usable for 16x16 semantic grids. The actual encode/decode
payload path remains the original explicit Huffman bitstream path.

## Prefix-Safe Top-k Correction

The first full 512-crop learned bitstream run exposed a critical correctness
issue:

```text
invalid run:
  20260628_stage3_b16_topk512_tf384_semposleft_g4_b4_gencodec512_no_pp_perceptual

symptom:
  roundtrip_failure_count: 10
  all_semantic_tokens_roundtrip: false
  all_detail_codes_roundtrip: true

cause:
  encoder used a vectorized teacher-forced top-k schedule, while decoder used
  sequential prefix replay. Small top-k boundary differences are enough to make
  an entropy-coded semantic stream invalid.

decision:
  do not use this run for paper tables, ablations, or external comparison.
```

Fix:

```text
src/coserdic/models/token_prior.py
  decoder_schedule_topk_indices:
    kept for fitting/diagnostics only
  decoder_prefix_topk_indices:
    new decoder-safe encoder schedule for real bitstream payloads

scripts/eval_stage2_learned_topk_escape_bitstream.py
scripts/eval_stage3_uniform_residual_bitstream.py
scripts/eval_stage3_residual_autoencoder_bitstream.py
  use decoder_prefix_topk_indices for actual compress/decompress evaluation

tests:
  tests/test_token_prior.py
  tests/test_topk_escape.py
```

Verification:

```text
.venv/bin/python -m pytest -q \
  tests/test_token_prior.py \
  tests/test_topk_escape.py \
  tests/test_bitstream.py \
  tests/test_eval_protocols.py

result:
  22 passed before the summarize-script additions
  26 passed after the summarize-script additions
```

Future summary files also include:

```text
semantic_topk_schedule: prefix_replay_decoder_safe
```

## Strict 512 GenCodec-Split Evaluation

No postprocess, prefix-safe:

```text
run:
  20260628_stage3_b16_topk512_tf384_semposleft_g4_b4_gencodec512_no_pp_perceptual_prefixsafe
wandb:
  wandb/offline-run-20260628_033123-fwip1kgx
summary:
  results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk512_tf384_semposleft_g4_b4_gencodec512_no_pp_perceptual_prefixsafe/summary.json
per-image:
  results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk512_tf384_semposleft_g4_b4_gencodec512_no_pp_perceptual_prefixsafe/per_image_metrics.jsonl

protocol:
  Kodak 24
  CLIC2020 test 428
  DIV2K val 100
  crop_size: 512
  count_status: all ok

roundtrip:
  semantic: true
  detail: true
  failure_count: 0

actual_payload_bpp / paper_bpp: 0.014225
semantic/detail bpp:              0.009223 / 0.005002
semantic_topk_hit:                0.768611
PSNR / MS-SSIM:                   21.996491 / 0.734915
LPIPS / DISTS:                    0.579664 / 0.355349
```

Unsharp3x3 strength 0.2, prefix-safe:

```text
run:
  20260628_stage3_b16_topk512_tf384_semposleft_g4_b4_gencodec512_pp_unsharp200_perceptual_prefixsafe
wandb:
  wandb/offline-run-20260628_040052-8e0bj6rz
summary:
  results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk512_tf384_semposleft_g4_b4_gencodec512_pp_unsharp200_perceptual_prefixsafe/summary.json
per-image:
  results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk512_tf384_semposleft_g4_b4_gencodec512_pp_unsharp200_perceptual_prefixsafe/per_image_metrics.jsonl

roundtrip:
  semantic: true
  detail: true
  failure_count: 0

actual_payload_bpp / paper_bpp: 0.014225
semantic/detail bpp:              0.009223 / 0.005002
semantic_topk_hit:                0.768611
PSNR / MS-SSIM:                   21.989531 / 0.734826
LPIPS / DISTS:                    0.576833 / 0.353773
delta vs semantic-only:
  PSNR:    +0.429216
  MS-SSIM: +0.004553
  LPIPS:   -0.001759
  DISTS:   -0.004716
```

Dataset-group summaries were written with `gencodec_reproduction` grouping:

```text
results/analysis/per_image_metric_summaries/20260628_stage3_b16_topk512_gencodec512_no_pp_prefixsafe_gencodec_groups.json
results/analysis/per_image_metric_summaries/20260628_stage3_b16_topk512_gencodec512_unsharp200_prefixsafe_gencodec_groups.json
```

Per-split unsharp3x3 strength 0.2:

```text
Kodak 24:
  actual_payload_bpp: 0.014320
  semantic/detail bpp: 0.009574 / 0.004747
  PSNR / MS-SSIM: 21.664208 / 0.722544
  LPIPS / DISTS: 0.637615 / 0.373392
  LPIPS / DISTS delta vs semantic-only: -0.003382 / -0.004944

CLIC2020 test 428:
  actual_payload_bpp: 0.013967
  semantic/detail bpp: 0.009009 / 0.004958
  PSNR / MS-SSIM: 22.358168 / 0.745362
  LPIPS / DISTS: 0.562502 / 0.350152
  LPIPS / DISTS delta vs semantic-only: -0.001330 / -0.004682

DIV2K val 100:
  actual_payload_bpp: 0.015306
  semantic/detail bpp: 0.010057 / 0.005249
  PSNR / MS-SSIM: 20.489843 / 0.692677
  LPIPS / DISTS: 0.623585 / 0.364563
  LPIPS / DISTS delta vs semantic-only: -0.003206 / -0.004809
```

No-postprocess to unsharp3x3 strength 0.2 image-wise comparison:

```text
artifacts:
  results/analysis/per_image_metric_summaries/20260628_stage3_b16_topk512_gencodec512_unsharp200_vs_no_pp_lpips.json
  results/analysis/per_image_metric_summaries/20260628_stage3_b16_topk512_gencodec512_unsharp200_vs_no_pp_dists.json
  results/analysis/per_image_metric_summaries/20260628_stage3_b16_topk512_gencodec512_unsharp200_vs_no_pp_psnr.json
  results/analysis/per_image_metric_summaries/20260628_stage3_b16_topk512_gencodec512_unsharp200_vs_no_pp_msssim.json

same actual_payload_bpp: true
LPIPS improvement count: 532 / 552
DISTS improvement count: 525 / 552
PSNR improvement count:  5 / 552
MS-SSIM improvement count: 85 / 552
```

512 top-k1024/top-k2048 fitting:

```text
topk1024:
  run:
    20260628_stage2_learned_topk1024_escape_huffman_fit_32768tokens_crop512_stage1_rateprior0005_b16_tf384_l6_12k
  wandb:
    wandb/offline-run-20260628_040934-tnc943ti
  prior:
    outputs/stage2_learned_entropy/20260628_stage2_learned_topk1024_escape_huffman_fit_32768tokens_crop512_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json
  calibration topk_hit_rate: 0.881559
  calibration escape_rate: 0.118441
  calibration mean_unpadded_bits_per_token: 9.058416
  calibration mean_payload_bpp_image_size: 0.008859

topk2048:
  run:
    20260628_stage2_learned_topk2048_escape_huffman_fit_32768tokens_crop512_stage1_rateprior0005_b16_tf384_l6_12k
  wandb:
    wandb/offline-run-20260628_041347-at87pklh
  prior:
    outputs/stage2_learned_entropy/20260628_stage2_learned_topk2048_escape_huffman_fit_32768tokens_crop512_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json
  calibration topk_hit_rate: 0.949401
  calibration escape_rate: 0.050599
  calibration mean_unpadded_bits_per_token: 8.971014
  calibration mean_payload_bpp_image_size: 0.008774
```

Topk2048 strict 512 full evaluation:

```text
run:
  20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_gencodec512_pp_unsharp200_perceptual_prefixsafe
wandb:
  wandb/offline-run-20260628_044635-3y540szk
summary:
  results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_gencodec512_pp_unsharp200_perceptual_prefixsafe/summary.json
per-image:
  results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_gencodec512_pp_unsharp200_perceptual_prefixsafe/per_image_metrics.jsonl

roundtrip:
  semantic: true
  detail: true
  failure_count: 0

actual_payload_bpp / paper_bpp: 0.013999
semantic/detail bpp:              0.008997 / 0.005002
semantic_topk_hit:                0.935108
PSNR / MS-SSIM:                   21.989628 / 0.734834
LPIPS / DISTS:                    0.576842 / 0.353785
delta vs semantic-only:
  PSNR:    +0.429277
  MS-SSIM: +0.004556
  LPIPS:   -0.001750
  DISTS:   -0.004722
```

Dataset-group summary:

```text
results/analysis/per_image_metric_summaries/20260628_stage3_b16_topk2048_gencodec512_unsharp200_prefixsafe_gencodec_groups.json
```

Topk2048 per-split unsharp3x3 strength 0.2:

```text
Kodak 24:
  actual_payload_bpp: 0.014121
  semantic/detail bpp: 0.009374 / 0.004747
  semantic_topk_hit: 0.914063
  LPIPS / DISTS: 0.637615 / 0.373392

CLIC2020 test 428:
  actual_payload_bpp: 0.013747
  semantic/detail bpp: 0.008788 / 0.004958
  semantic_topk_hit: 0.940913
  LPIPS / DISTS: 0.562515 / 0.350167

DIV2K val 100:
  actual_payload_bpp: 0.015052
  semantic/detail bpp: 0.009802 / 0.005250
  semantic_topk_hit: 0.915313
  LPIPS / DISTS: 0.623577 / 0.364565
```

Topk512 to topk2048 comparison:

```text
artifacts:
  results/analysis/per_image_metric_summaries/20260628_stage3_b16_gencodec512_topk2048_vs_topk512_unsharp200_lpips.json
  results/analysis/per_image_metric_summaries/20260628_stage3_b16_gencodec512_topk2048_vs_topk512_unsharp200_dists.json

overall:
  actual_payload_bpp: 0.014224840 -> 0.013999276
  semantic_payload_bpp: 0.009223330 -> 0.008997378
  detail_payload_bpp: 0.005001510 -> 0.005001897
  semantic_topk_hit_rate: 0.768611 -> 0.935108

rate win count:
  550 / 552 images

quality:
  stage3 LPIPS delta vs topk512: +0.000009
  stage3 DISTS delta vs topk512: +0.000012
  This is numerically negligible and expected because the decoded semantic
  tokens and detail latents roundtrip to the same image; top-k only changes the
  transmitted entropy stream.
```

Strict 512 reconstruction export and distribution metrics:

```text
no-postprocess perceptual eval:
  run:
    20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_gencodec512_no_pp_perceptual_prefixsafe
  wandb:
    wandb/offline-run-20260628_063055-h1eim4fe
  summary:
    results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_gencodec512_no_pp_perceptual_prefixsafe/summary.json

no-postprocess reconstruction export:
  run:
    20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_gencodec512_no_pp_recon_export_prefixsafe
  summary:
    results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_gencodec512_no_pp_recon_export_prefixsafe/summary.json

unsharp0.2 reconstruction export:
  run:
    20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_gencodec512_pp_unsharp200_recon_export_prefixsafe
  wandb:
    wandb/offline-run-20260628_055411-jgy2fz74
  summary:
    results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_gencodec512_pp_unsharp200_recon_export_prefixsafe/summary.json

unsharp0.15 reconstruction export:
  run:
    20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_gencodec512_pp_unsharp150_recon_export_prefixsafe
  wandb:
    wandb/offline-run-20260628_081439-v4eqntun
  summary:
    results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_gencodec512_pp_unsharp150_recon_export_prefixsafe/summary.json

shared:
  protocol: CoD-style 512-crop reproduction composition
  data: Kodak 24 + CLIC2020 test 428 + DIV2K val 100
  crop: 512 x 512 center crop
  actual_payload_bpp / paper_bpp: 0.013999
  semantic/detail bpp: 0.008997 / 0.005002
  semantic_topk_hit: 0.935108
  roundtrip failure_count: 0
```

Postprocess tradeoff on strict 512:

```text
artifact:
  results/analysis/stage3_topk2048_gencodec512_postprocess_tradeoff.json

metric note:
  patch128 values below are torch-fidelity single-grid diagnostics
  (`--patch-size 128 --patch-stride 128`). They are not CoD/CoD-Lite
  dataset-level patch FID.
  CoD / CoD-Lite patch FID must be reported per dataset with exact patch
  size, shift count, and backend. Treat CLIC patch128 as a legacy diagnostic
  unless explicitly reproducing a paper-specific setting; the public
  GenCodec/CoD-Lite metric scripts default to patch256 and CoD-Lite README
  states Kodak512=64 and other datasets=256.

no-postprocess:
  PSNR / MS-SSIM:      21.996587 / 0.734923
  LPIPS / DISTS:       0.579672 / 0.355362
  image FID / KID:     180.091482 / 0.061146
  patch128_s128 diagnostic FID / KID:  161.001764 / 0.108373

unsharp3x3 strength 0.2:
  PSNR / MS-SSIM:      21.989628 / 0.734834
  LPIPS / DISTS:       0.576842 / 0.353785
  image FID / KID:     180.463999 / 0.061824
  patch128_s128 diagnostic FID / KID:  161.242647 / 0.108373

unsharp3x3 strength 0.15:
  PSNR / MS-SSIM:      21.991413 / 0.734857
  LPIPS / DISTS:       0.577543 / 0.354169
  image FID / KID:     180.385064 / 0.061658
  patch128_s128 diagnostic FID / KID:  161.178728 / 0.108383

unsharp0.2 minus no-postprocess:
  bpp:       +0.000000
  PSNR:      -0.006959
  MS-SSIM:   -0.000089
  LPIPS:     -0.002830
  DISTS:     -0.001577
  image FID: +0.372518
  image KID: +0.000678
  patch FID: +0.240883
  patch KID: approximately tied

per-image wins by unsharp0.2:
  LPIPS:    532 / 552
  DISTS:    525 / 552
  PSNR:       5 / 552
  MS-SSIM:   85 / 552
```

CoD-Lite / GLC / DLF-style patch FID/KID, no-postprocess, strict 512,
torch-fidelity backend:

```text
artifacts:
  results/analysis/image_distribution_metrics/20260628_stage3_b16_topk2048_no_pp_gencodec512_kodak_gencodec_patch64_n2_fid_kid.json
  results/analysis/image_distribution_metrics/20260628_stage3_b16_topk2048_no_pp_gencodec512_clic2020_test_gencodec_patch256_n2_fid_kid.json
  results/analysis/image_distribution_metrics/20260628_stage3_b16_topk2048_no_pp_gencodec512_div2k_val_gencodec_patch256_n2_fid_kid.json

Kodak 24:
  patch protocol: 64px non-overlap + 32px half-shift, 2712 patches
  FID / KID:      216.298774 / 0.150770

CLIC2020 test 428:
  patch protocol: 256px non-overlap + 128px half-shift, 2140 patches
  FID / KID:      152.007812 / 0.079693

DIV2K val 100:
  patch protocol: 256px non-overlap + 128px half-shift, 500 patches
  FID / KID:      234.593242 / 0.092560

backend caveat:
  These values align the non-Kodak patch size used by CoD-Lite/GLC/DLF README
  scripts, but they are not the CoD paper CLIC128 setting. They also use
  torch-fidelity; GenCodec official scripts use torchmetrics FID, so external
  paper-table numbers should label the backend.
```

Legacy CLIC128 patch FID diagnostic, strict 512, torchmetrics backend:

```text
artifact:
  results/analysis/stage3_topk2048_cod512_torchmetrics_fid_summary.json

per-run artifacts:
  results/analysis/image_distribution_metrics/20260628_stage3_b16_topk2048_no_pp_cod512_kodak_patch64_n2_torchmetrics_fid.json
  results/analysis/image_distribution_metrics/20260628_stage3_b16_topk2048_unsharp150_cod512_kodak_patch64_n2_torchmetrics_fid.json
  results/analysis/image_distribution_metrics/20260628_stage3_b16_topk2048_unsharp200_cod512_kodak_patch64_n2_torchmetrics_fid.json
  results/analysis/image_distribution_metrics/20260628_stage3_b16_topk2048_no_pp_cod512_clic2020_test_patch128_n2_torchmetrics_fid.json
  results/analysis/image_distribution_metrics/20260628_stage3_b16_topk2048_unsharp150_cod512_clic2020_test_patch128_n2_torchmetrics_fid.json
  results/analysis/image_distribution_metrics/20260628_stage3_b16_topk2048_unsharp200_cod512_clic2020_test_patch128_n2_torchmetrics_fid.json

Kodak 24:
  patch protocol: 64px non-overlap + 32px half-shift, 2712 patches
  no-postprocess FID: 216.777847
  unsharp0.15 FID:    216.593414  delta -0.184433
  unsharp0.20 FID:    216.635513  delta -0.142334

CLIC2020 test 428:
  patch protocol: 128px non-overlap + 64px half-shift, 10700 patches
  no-postprocess FID: 156.727158
  unsharp0.15 FID:    156.717102  delta -0.010056
  unsharp0.20 FID:    156.715881  delta -0.011276

interpretation:
  Under the CoD paper-style torchmetrics FID protocol, both unsharp strengths
  are slightly better than no-postprocess on Kodak64 and CLIC128. The absolute
  deltas are tiny, so this supports postprocess as a harmless perceptual
  tuning knob rather than as a core distribution-quality breakthrough.
```

Unsharp-strength sweep:

```text
artifact:
  results/analysis/stage3_topk2048_gencodec512_unsharp_strength_sweep.json

strength 0.0:
  LPIPS / DISTS: 0.579672 / 0.355362
  PSNR / MS-SSIM: 21.996587 / 0.734923
  image FID / patch FID: 180.091482 / 161.001764

strength 0.1:
  LPIPS / DISTS: 0.578248 / 0.354560
  PSNR / MS-SSIM: 21.993168 / 0.734879
  LPIPS / DISTS wins vs none: 532 / 525

strength 0.15:
  LPIPS / DISTS: 0.577543 / 0.354169
  PSNR / MS-SSIM: 21.991413 / 0.734857
  image FID / patch FID: 180.385064 / 161.178728
  LPIPS / DISTS wins vs none: 533 / 526

strength 0.2:
  LPIPS / DISTS: 0.576842 / 0.353785
  PSNR / MS-SSIM: 21.989628 / 0.734834
  image FID / patch FID: 180.463999 / 161.242647
  LPIPS / DISTS wins vs none: 532 / 525
```

Fixed detail-gain sweep at unsharp3x3 strength 0.15:

```text
artifacts:
  results/analysis/stage3_topk2048_cod512_unsharp150_gain_sweep_summary.json
  results/analysis/stage3_topk2048_cod512_unsharp150_gain_sweep_per_image.json

gain 1.00:
  run:
    20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_gencodec512_pp_unsharp150_perceptual_prefixsafe
  actual_payload_bpp: 0.013999
  PSNR / MS-SSIM:      21.991413 / 0.734857
  LPIPS / DISTS:       0.577543 / 0.354169
  Kodak64 / CLIC128 CoD-style torchmetrics FID:
    216.593414 / 156.717102

gain 0.75:
  run:
    20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_cod512_gain075_pp_unsharp150_perceptual_recon
  wandb:
    wandb/offline-run-20260628_100503-tbu2gp28
  actual_payload_bpp: 0.013999
  PSNR / MS-SSIM:      21.965810 / 0.734971
  LPIPS / DISTS:       0.574178 / 0.354746
  Kodak64 / CLIC128 CoD-style torchmetrics FID:
    215.559616 / 153.337830
  per-image wins vs gain1.00:
    LPIPS 495 / 552, DISTS 180 / 552, PSNR 50 / 552, MS-SSIM 279 / 552

gain 0.625:
  run:
    20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_cod512_gain0625_pp_unsharp150_perceptual_recon
  wandb:
    wandb/offline-run-20260628_111950-rgg7tc4w
  actual_payload_bpp: 0.013999
  PSNR / MS-SSIM:      21.929778 / 0.734715
  LPIPS / DISTS:       0.573180 / 0.355106
  Kodak64 / CLIC128 CoD-style torchmetrics FID:
    215.246185 / 152.066864
  per-image wins vs gain1.00:
    LPIPS 481 / 552, DISTS 170 / 552, PSNR 30 / 552, MS-SSIM 224 / 552

gain 0.50:
  run:
    20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_cod512_gain050_pp_unsharp150_perceptual_recon
  wandb:
    wandb/offline-run-20260628_104119-jbua04d0
  actual_payload_bpp: 0.013999
  PSNR / MS-SSIM:      21.879395 / 0.734248
  LPIPS / DISTS:       0.572700 / 0.355515
  Kodak64 / CLIC128 CoD-style torchmetrics FID:
    215.036667 / 150.943024
  per-image wins vs gain1.00:
    LPIPS 460 / 552, DISTS 159 / 552, PSNR 18 / 552, MS-SSIM 174 / 552

decision note:
  detail_gain is a fixed decoder-side operating-point setting in these runs,
  not image-specific side information, so actual_payload_bpp is unchanged.
  If a future method transmits per-image or spatially varying gain/control, it
  must be counted in actual_payload_bpp.
```

Interpretation:

```text
strict 512 path:
  now valid for actual_payload_bpp/paper_bpp reporting because semantic and
  detail streams roundtrip exactly under decoder prefix replay.

rate point:
  topk2048 is the current strict-512 semantic entropy leader and brings the
  main unsharp perceptual point just below 0.014 actual_payload_bpp.

perceptual point:
  gain0.50 + unsharp3x3 strength 0.15 is now the strongest LPIPS and CoD-style
  patch-FID point checked at the strict 512 rate. It sacrifices PSNR and DISTS,
  so it should be reported as an LPIPS/FID-first operating point.

balanced point:
  gain0.625 + unsharp3x3 strength 0.15 is the current LPIPS/FID-balanced point.
  It preserves most of the gain0.50 LPIPS/FID improvement while recovering
  some PSNR/MS-SSIM/DISTS. gain0.75 is the MS-SSIM-friendlier reduced-gain
  point.

fidelity point:
  no-postprocess remains the safest PSNR point. For DISTS-first reporting,
  gain1.00 + unsharp3x3 strength 0.20 is still strongest among the checked
  postprocess variants. CoD paper-style patch FID now clearly prefers reduced
  detail gain, so Stage 3 should present multiple operating points rather than
  collapse all perceptual/fidelity behavior into one number.

external SOTA claim:
  still not justified. We have a valid 512 protocol point at extremely low bpp,
  but external official-codec comparison, human/visual audits, and stronger
  reproduced baselines are still open.
```

## Strict 512 Failure-Mode Audit

Failure-mode analysis artifacts:

```text
script:
  scripts/analyze_stage3_failure_modes.py

json artifacts:
  results/analysis/stage3_failure_modes/20260628_stage3_b16_topk2048_cod512_gain050_unsharp150_failure_modes.json
  results/analysis/stage3_failure_modes/20260628_stage3_b16_topk2048_cod512_gain0625_unsharp150_failure_modes.json
  results/analysis/stage3_failure_modes/20260628_stage3_b16_topk2048_cod512_gain075_unsharp150_failure_modes.json

visual galleries:
  results/analysis/stage3_visual_galleries/20260628_stage3_b16_topk2048_cod512_gain050_unsharp150_worst_lpips.md
  results/analysis/stage3_visual_galleries/20260628_stage3_b16_topk2048_cod512_gain050_unsharp150_worst_dists.md
  results/analysis/stage3_visual_galleries/20260628_stage3_b16_topk2048_cod512_gain050_unsharp150_lowest_psnr.md
  results/analysis/stage3_visual_galleries/20260628_stage3_b16_topk2048_cod512_gain0625_unsharp150_lpips_regressions.md
```

Semantic-only to Stage 3 deltas at strict 512, topk2048, unsharp0.15:

```text
gain0.50:
  PSNR mean delta:    +0.319094, improvements 552 / 552
  MS-SSIM mean delta: +0.003972, improvements 549 / 552
  LPIPS mean delta:   -0.005892, improvements 516 / 552
  DISTS mean delta:   -0.002990, improvements 525 / 552

gain0.625:
  PSNR mean delta:    +0.369476, improvements 552 / 552
  MS-SSIM mean delta: +0.004438, improvements 547 / 552
  LPIPS mean delta:   -0.005412, improvements 477 / 552
  DISTS mean delta:   -0.003399, improvements 516 / 552

gain0.75:
  PSNR mean delta:    +0.405508, improvements 552 / 552
  MS-SSIM mean delta: +0.004694, improvements 543 / 552
  LPIPS mean delta:   -0.004414, improvements 435 / 552
  DISTS mean delta:   -0.003759, improvements 506 / 552
```

Overall driver correlations for gain0.50:

```text
LPIPS:
  semantic_payload_bpp:  +0.575
  actual_payload_bpp:    +0.517
  semantic_topk_hit:     -0.375
  residual_abs_mean:     +0.262

DISTS:
  residual_abs_mean:     +0.286
  actual_payload_bpp:    +0.278
  semantic_payload_bpp:  +0.240

PSNR:
  actual_payload_bpp:    -0.749
  detail_entropy_bits:   -0.667
  residual_std:          -0.619
  semantic_payload_bpp:  -0.619

MS-SSIM:
  actual_payload_bpp:    -0.734
  semantic_payload_bpp:  -0.683
  residual_abs_mean:     -0.523
  semantic_topk_hit:     +0.513
```

Visual audit:

```text
dominant failure class:
  highly structured high-frequency content: branches, leaves, grids, repeated
  thin lines, fabric-like or building-like lattices.

observed behavior:
  semantic-only reconstructions preserve coarse color/layout but smear or
  hallucinate fine structure.
  the current 4-bit static detail residual improves fidelity metrics but cannot
  reconstruct dense high-frequency geometry at this payload.
  reduced detail gain improves LPIPS and CoD-style patch FID because it avoids
  over-committing to weak residual texture, but it cannot solve the missing
  structure problem.

Stage 4 / 5 implication:
  the next large-improvement attempt should be model-side, not more fixed
  postprocess tuning. The most plausible Core-MVP-preserving direction is a
  CoD-Lite / CoD diffusion backbone conditioned only on decoded semantic tokens,
  decoded detail residual, and decoder-reproducible auxiliary features, with no
  image-specific side information unless counted in actual_payload_bpp. It
  should target structured texture regeneration and keep the semantic/detail
  entropy split.
```

## Open Items

```text
1. Continue decoder sweep on the strict 512 path:
   done: unsharp strength 0.0, 0.1, 0.15, 0.2 at detail_gain 1.0;
         detail_gain 0.5, 0.625, and 0.75 at unsharp0.15
   next: stop fixed postprocess/gain hand-tuning for now and move model-side
         improvements.
   goal: separate LPIPS/FID-first, balanced, DISTS-first, and PSNR-first
         operating points without changing actual_payload_bpp.
2. Compare against refreshed official baselines or reproduced metrics before
   making any external "large improvement" claim.
3. Visual audits / failure galleries from strict 512 reconstructions:
   done for gain0.50 and gain0.625.
   next: use these galleries to design the active CoD-Lite / CoD Stage 4
         diffusion adapter experiment in
         `021_stage4_cod_codlite_parallel_backbone_policy.md`.
         The earlier ResUNet decoder-refiner path is archived in
         `020_stage4_decoder_side_refiner.md` as engineering diagnostics only
         and must not steer the MVP Stage 4 architecture.
4. Keep all paper tables on actual_payload_bpp and exclude failed
   teacher-forced-topk bitstream runs.
```

## Current Working Judgment

The batch-16 rateprior branch is not yet a claim of large improvement over all
published codecs. It is, however, the strongest internal Core MVP
rate-perception branch so far. The 256-crop CoSER common path beats the previous
ft500 topk2048 internal branch on both actual payload bpp and LPIPS/DISTS, and
the strict 512-crop GenCodec-split path now has a valid prefix-safe
actual-payload result with distribution diagnostics:

```text
reason:
  It keeps the CoSER-DiC novelty axis intact.
  It improves semantic and Stage 3 perceptual quality.
  It uses actual entropy-coded payload bpp, not estimated bpp.
  It roundtrips exactly for both semantic and detail streams.
  The b16 topk2048 reconnection closes the previous ft500 topk2048 bpp gap on
  CoSER common while improving LPIPS/DISTS.
  Strict 512-crop GenCodec-split evaluation is now functional with a separate
  context_length=256 prior and 16x16 residual prior.
  Strict 512 topk2048 lowers the unsharp perceptual point to
  0.013999 actual_payload_bpp with exact semantic/detail roundtrip.
  Strict 512 gain0.50 + unsharp0.15 is the strongest LPIPS and CoD-style
  patch-FID point checked so far; gain0.625 + unsharp0.15 is the current
  LPIPS/FID-balanced reduced-gain point; gain1.00 + unsharp0.20 remains the
  DISTS-first point.
  Failure-mode analysis shows the remaining weakness is structured
  high-frequency content, which should be addressed by the active CoD-Lite /
  CoD diffusion-backbone integration rather than by more hand-tuned
  postprocess or the archived ResUNet refiner branch.
  Initial ResUNet decoder-refiner implementation exists only as archived
  engineering diagnostics: it validated unchanged-payload model-side evaluation
  plumbing, but it is not a promoted Stage 4 result and should not steer the
  active CoD-Lite / CoD architecture.
  External reproduced baselines remain necessary before any large-improvement
  claim.
```
