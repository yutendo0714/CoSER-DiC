# Stage 3 Residual Progress

Date: 2026-06-27 JST

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

Decision: promote position Huffman to the active Stage 3 residual entropy path.
The gain is small, especially for b5, but it is consistent across Kodak,
DIV2K, and CLIC while preserving the same decoded residual grid. The active
quality bootstrap is now d32/b5/r0.25 position Huffman; the low-rate anchor is
d32/b4/r0.25 position Huffman. The older global Huffman results remain useful
as a simpler ablation.

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
all three checked datasets. The d32/b4/r0.25 position-Huffman setting is the
low-rate anchor; the d32/b5/r0.25 position-Huffman setting is the active quality
bootstrap because it improves Kodak, DIV2K, and CLIC while staying below 0.02
total payload bpp. The residual-grid clipping ratio is zero on Kodak, and the
code entropy is far below the fixed width. This suggests Stage 3 should start
simple:

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
3. Evaluate with total payload bpp and full stream bpp, not estimated bpp.
4. Keep d32/b5/r0.25 position Huffman as the quality bootstrap and
   d32/b4/r0.25 position Huffman as the low-rate anchor.
```
