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
all three checked datasets. The active anchors are now hybrid
position/semantic-position Huffman with smoothing=0: g8 for the d32/b4/r0.25
low-rate setting and g4 for the d32/b5/r0.25 quality setting. They preserve
the same decoded residual grids while reducing actual transmitted bytes. The
residual-grid clipping ratio is zero on Kodak, and the code entropy is far
below the fixed width. This suggests Stage 3 should start simple:

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
4. Keep d32/b5/r0.25 hybrid position/semantic-position g4 Huffman with
   smoothing=0 as the quality bootstrap and d32/b4/r0.25 hybrid
   position/semantic-position g8 Huffman with smoothing=0 as the low-rate
   anchor.
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
