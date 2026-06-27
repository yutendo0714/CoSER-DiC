# Stage 3 Uniform Residual Bootstrap

Date: 2026-06-27  
Status: Stage 3 actual-bitstream bootstrap baseline with static residual entropy coding  
Parent policy: `docs/research/design_decisions/003_official_implementation_reference_policy.md`

## Purpose

Stage 2 now has an active learned semantic-token stream with actual transmitted
bytes and exact token roundtrip. Stage 3 needs a detail stream that improves
fidelity without collapsing CoSER-DiC into a single monolithic pixel codec.

The first Stage 3 baseline is intentionally simple:

```text
decoded semantic reconstruction
  -> pixel residual
  -> low-resolution residual grid
  -> uniform quantization
  -> fixed-bit, zlib-fixed-bit, or static-Huffman payload
  -> CoSERBitstream detail_latents
  -> exact detail-code roundtrip
  -> residual upsample + semantic reconstruction
```

This is not the final detail model. It is an actual-byte baseline for the
semantic/detail split and a target for learned residual entropy coding.

## Implementation

```text
src/coserdic/entropy/residual_grid.py
scripts/fit_stage3_residual_huffman_prior.py
scripts/eval_stage3_uniform_residual_bitstream.py
tests/test_residual_grid.py
```

The active semantic stream is still:

```text
outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json
```

The Stage 3 eval script uses the same decoder-schedule top-k reconstruction as
Stage 2 and reports semantic payload bpp, detail payload bpp, total payload
bpp, full stream bpp, quality metrics, and both semantic/detail roundtrip. The
residual grid can be coded with fixed bits, zlib-fixed bits, or a decoder-known
static Huffman prior.

## Kodak Sweep

All runs used:

```text
checkpoint:
  checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt

semantic prior:
  outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json

detail codec:
  zlib_fixed_bits

roundtrip:
  semantic tokens: true
  detail codes: true
```

Kodak 24 center-crop results:

```text
d32 b4 r0.50:
  run: 20260627_stage3_uniform_residual_d32_b4_r050_zlib_kodak_eval
  wandb: wandb/offline-run-20260627_113114-t0adm0wh
  detail_payload_bpp: 0.006948
  total_payload_bpp: 0.017670
  PSNR delta: +0.2309 dB
  MS-SSIM delta: -0.00037

d32 b5 r0.50:
  run: 20260627_stage3_uniform_residual_d32_b5_r050_zlib_kodak_eval
  wandb: wandb/offline-run-20260627_113136-iv0y60uw
  detail_payload_bpp: 0.014125
  total_payload_bpp: 0.024846
  PSNR delta: +0.3721 dB
  MS-SSIM delta: +0.00472

d32 b6 r0.50:
  run: 20260627_stage3_uniform_residual_d32_b6_r050_zlib_kodak_eval
  wandb: wandb/offline-run-20260627_113047-gsnxosko
  detail_payload_bpp: 0.017298
  total_payload_bpp: 0.028020
  PSNR delta: +0.3962 dB
  MS-SSIM delta: +0.00575

d16 b4 r0.50:
  run: 20260627_stage3_uniform_residual_d16_b4_r050_zlib_kodak_eval
  wandb: wandb/offline-run-20260627_113158-r517tlsq
  detail_payload_bpp: 0.022578
  total_payload_bpp: 0.033300
  PSNR delta: +0.7412 dB
  MS-SSIM delta: +0.02689

d16 b3 r0.50:
  run: 20260627_stage3_uniform_residual_d16_b3_r050_zlib_kodak_eval
  wandb: wandb/offline-run-20260627_113219-9wv16q1j
  detail_payload_bpp: 0.024083
  total_payload_bpp: 0.034805
  PSNR delta: -0.0394 dB
  MS-SSIM delta: -0.01056

d32 b4 r0.25:
  run: 20260627_stage3_uniform_residual_d32_b4_r025_zlib_kodak_eval_stats
  wandb: wandb/offline-run-20260627_113714-g5te9p9x
  detail_payload_bpp: 0.008759
  total_payload_bpp: 0.019480
  full_stream_bpp: 0.080907
  PSNR: 21.1068 dB
  MS-SSIM: 0.68042
  PSNR delta: +0.3705 dB
  MS-SSIM delta: +0.00465
  residual_grid_abs_mean: 0.02022
  residual_grid_std: 0.02511
  residual_grid_clip_ratio: 0.0
  detail_code_entropy_bits: 1.6878

d16 b4 r0.25:
  run: 20260627_stage3_uniform_residual_d16_b4_r025_zlib_kodak_eval
  wandb: wandb/offline-run-20260627_113308-xwm4ijhh
  detail_payload_bpp: 0.031204
  total_payload_bpp: 0.041926
  PSNR delta: +0.8644 dB
  MS-SSIM delta: +0.03481
```

## Cross-Dataset Check

Selected low-bpp candidate:

```text
detail_downsample_factor: 32
detail_bits: 4
detail_range: 0.25
detail_codec: zlib_fixed_bits
```

Results:

```text
DIV2K first 100:
  run: 20260627_stage3_uniform_residual_d32_b4_r025_zlib_div2k100_eval
  wandb: wandb/offline-run-20260627_113413-s5ernjrm
  semantic_payload_bpp: 0.010425
  detail_payload_bpp: 0.008601
  total_payload_bpp: 0.019026
  full_stream_bpp: 0.080448
  PSNR delta: +0.3055 dB
  MS-SSIM delta: +0.00403
  roundtrip_failure_count: 0

CLIC professional valid:
  run: 20260627_stage3_uniform_residual_d32_b4_r025_zlib_clicval64_eval
  wandb: wandb/offline-run-20260627_113437-rybb6lqc
  num_images: 41
  semantic_payload_bpp: 0.009906
  detail_payload_bpp: 0.008089
  total_payload_bpp: 0.017995
  full_stream_bpp: 0.079405
  PSNR delta: +0.5649 dB
  MS-SSIM delta: +0.00246
  roundtrip_failure_count: 0
```

## Static Residual Huffman Update

The d32/b4/r0.25 residual symbols have low entropy, so a decoder-known static
Huffman prior was fit from OpenImages + DIV2K calibration images:

```text
run: 20260627_stage3_residual_huffman_d32_b4_r025_4096calib_oi_div2k
wandb: wandb/offline-run-20260627_114352-rrbpxgyp

prior:
  outputs/stage3_residual_entropy/20260627_stage3_residual_huffman_d32_b4_r025_4096calib_oi_div2k/static_residual_grid_huffman_prior.json

num_images: 4096
symbol_entropy_bits: 1.9339
mean_huffman_bits_per_symbol: 2.0087
fixed_bits_per_symbol: 4
clip_ratio: 0.000136
```

Actual-byte evaluation with the same residual values and Huffman detail payload:

```text
Kodak 24:
  run: 20260627_stage3_residual_huffman_d32_b4_r025_4096calib_kodak_eval
  wandb: wandb/offline-run-20260627_114414-j6e6y0la
  semantic_payload_bpp: 0.010722
  detail_payload_bpp: 0.005544
  total_payload_bpp: 0.016266
  full_stream_bpp: 0.077693
  PSNR delta: +0.3705 dB
  MS-SSIM delta: +0.00465
  roundtrip_failure_count: 0

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
  num_images: 41
  semantic_payload_bpp: 0.009906
  detail_payload_bpp: 0.005443
  total_payload_bpp: 0.015348
  full_stream_bpp: 0.076758
  PSNR delta: +0.5649 dB
  MS-SSIM delta: +0.00246
  roundtrip_failure_count: 0
```

Relative to the zlib-fixed-bit d32/b4/r0.25 bootstrap, static Huffman keeps the
same reconstruction quality and reduces Kodak detail payload:

```text
zlib detail_payload_bpp: 0.008759
Huffman detail_payload_bpp: 0.005544
delta: -0.003215 bpp

zlib total_payload_bpp: 0.019480
Huffman total_payload_bpp: 0.016266
delta: -0.003215 bpp
```

## Static-Huffman RD Probe

The static-Huffman path changed the rate tradeoff enough that nearby residual
quantizers were rechecked with actual transmitted bytes:

```text
Kodak d32 b4 r0.50:
  run: 20260627_stage3_residual_huffman_d32_b4_r050_4096calib_kodak_eval
  wandb: wandb/offline-run-20260627_115347-ebo6u4pi
  detail_payload_bpp: 0.004425
  total_payload_bpp: 0.015147
  full_stream_bpp: 0.076574
  PSNR delta: +0.2309 dB
  MS-SSIM delta: -0.00037
  roundtrip_failure_count: 0

Kodak d32 b5 r0.25:
  run: 20260627_stage3_residual_huffman_d32_b5_r025_4096calib_kodak_eval
  wandb: wandb/offline-run-20260627_115138-kbyixk0n
  detail_payload_bpp: 0.008286
  total_payload_bpp: 0.019007
  full_stream_bpp: 0.080434
  PSNR delta: +0.3957 dB
  MS-SSIM delta: +0.00576
  roundtrip_failure_count: 0

Kodak d32 b6 r0.25:
  run: 20260627_stage3_residual_huffman_d32_b6_r025_4096calib_kodak_eval
  wandb: wandb/offline-run-20260627_115243-2h5dhyss
  detail_payload_bpp: 0.011256
  total_payload_bpp: 0.021978
  full_stream_bpp: 0.083420
  PSNR delta: +0.4015 dB
  MS-SSIM delta: +0.00605
  roundtrip_failure_count: 0
```

The b5/r0.25 candidate was cross-checked:

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
  num_images: 41
  semantic_payload_bpp: 0.009906
  detail_payload_bpp: 0.007970
  total_payload_bpp: 0.017876
  full_stream_bpp: 0.079286
  PSNR delta: +0.7171 dB
  MS-SSIM delta: +0.00370
  roundtrip_failure_count: 0
```

## Compact Full-Stream Cross-Dataset Check

The active and low-rate position-Huffman settings were re-evaluated with the
compact CoSERBitstream header. Payload bpp and reconstruction are unchanged;
only the transmitted container overhead changes.

```text
active d32 b5 r0.25 position Huffman:
  Kodak:
    run: 20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_kodak_eval_compacthdr
    wandb: wandb/offline-run-20260627_123512-dq2t1e7v
    detail_payload_bpp: 0.008250
    total_payload_bpp: 0.018972
    compact_full_stream_bpp: 0.037282
    PSNR delta: +0.3957 dB
    MS-SSIM delta: +0.00576
  DIV2K first 100:
    run: 20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_div2k100_eval_compacthdr
    wandb: wandb/offline-run-20260627_124113-wigh652w
    detail_payload_bpp: 0.008549
    total_payload_bpp: 0.018973
    compact_full_stream_bpp: 0.037284
    PSNR delta: +0.4601 dB
    MS-SSIM delta: +0.00537
  CLIC professional valid:
    run: 20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_clicval64_eval_compacthdr
    wandb: wandb/offline-run-20260627_124142-lsatlbwp
    num_images: 41
    detail_payload_bpp: 0.007967
    total_payload_bpp: 0.017873
    compact_full_stream_bpp: 0.036183
    PSNR delta: +0.7171 dB
    MS-SSIM delta: +0.00370

low-rate d32 b4 r0.25 position Huffman:
  Kodak:
    run: 20260627_stage3_residual_poshuff_d32_b4_r025_4096calib_kodak_eval_compacthdr
    wandb: wandb/offline-run-20260627_123537-95rgqsve
    detail_payload_bpp: 0.005483
    total_payload_bpp: 0.016205
    compact_full_stream_bpp: 0.034515
    PSNR delta: +0.3705 dB
    MS-SSIM delta: +0.00465
  DIV2K first 100:
    run: 20260627_stage3_residual_poshuff_d32_b4_r025_4096calib_div2k100_eval_compacthdr
    wandb: wandb/offline-run-20260627_124219-ugd3uztm
    detail_payload_bpp: 0.005837
    total_payload_bpp: 0.016262
    compact_full_stream_bpp: 0.034573
    PSNR delta: +0.3055 dB
    MS-SSIM delta: +0.00403
  CLIC professional valid:
    run: 20260627_stage3_residual_poshuff_d32_b4_r025_4096calib_clicval64_eval_compacthdr
    wandb: wandb/offline-run-20260627_124247-eryfeq3k
    num_images: 41
    detail_payload_bpp: 0.005267
    total_payload_bpp: 0.015172
    compact_full_stream_bpp: 0.033483
    PSNR delta: +0.5649 dB
    MS-SSIM delta: +0.00246

roundtrip_failure_count: 0 for all six compact-header runs
```

## CRC32 Compact-Header Update

The compact header originally used a 32-byte SHA256 checksum. For crop-level
low-bitrate experiments this fixed cost is visible in actual bpp. A compact
CRC32 checksum mode was added as an explicit stream option:

```text
src/coserdic/entropy/bitstream.py:
  CoSERBitstream(header_codec="compact", checksum_codec="crc32")

evaluation flags:
  --stream-header-codec compact
  --stream-checksum-codec crc32

tests:
  compact CRC32 roundtrip
  CRC32 corruption rejection
  SHA256 compact backward decode remains supported
```

CRC32 compact-header results:

```text
active d32 b5 r0.25 position Huffman:
  Kodak:
    run: 20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_kodak_eval_compacthdr_crc32
    wandb: wandb/offline-run-20260627_125032-8zoimefl
    total_payload_bpp: 0.018972
    compact_crc32_full_stream_bpp: 0.033864
    PSNR delta: +0.3957 dB
    MS-SSIM delta: +0.00576
  DIV2K first 100:
    run: 20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_div2k100_eval_compacthdr_crc32
    wandb: wandb/offline-run-20260627_125212-d6o802xx
    total_payload_bpp: 0.018973
    compact_crc32_full_stream_bpp: 0.033866
    PSNR delta: +0.4601 dB
    MS-SSIM delta: +0.00537
  CLIC professional valid:
    run: 20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_clicval64_eval_compacthdr_crc32
    wandb: wandb/offline-run-20260627_125238-0u6i8ugo
    total_payload_bpp: 0.017873
    compact_crc32_full_stream_bpp: 0.032765
    PSNR delta: +0.7171 dB
    MS-SSIM delta: +0.00370

low-rate d32 b4 r0.25 position Huffman:
  Kodak:
    run: 20260627_stage3_residual_poshuff_d32_b4_r025_4096calib_kodak_eval_compacthdr_crc32
    wandb: wandb/offline-run-20260627_125101-cpql516l
    total_payload_bpp: 0.016205
    compact_crc32_full_stream_bpp: 0.031097
    PSNR delta: +0.3705 dB
    MS-SSIM delta: +0.00465
  DIV2K first 100:
    run: 20260627_stage3_residual_poshuff_d32_b4_r025_4096calib_div2k100_eval_compacthdr_crc32
    wandb: wandb/offline-run-20260627_125317-h68rvzaf
    total_payload_bpp: 0.016262
    compact_crc32_full_stream_bpp: 0.031155
    PSNR delta: +0.3055 dB
    MS-SSIM delta: +0.00403
  CLIC professional valid:
    run: 20260627_stage3_residual_poshuff_d32_b4_r025_4096calib_clicval64_eval_compacthdr_crc32
    wandb: wandb/offline-run-20260627_125345-oo08oouo
    total_payload_bpp: 0.015172
    compact_crc32_full_stream_bpp: 0.030065
    PSNR delta: +0.5649 dB
    MS-SSIM delta: +0.00246

roundtrip_failure_count: 0 for all six CRC32 compact-header runs
```

## Short Codec-ID Update

The compact header still carried long experiment names such as
`stage3-uniform-residual-mvp`. The transmitted header now uses short registry
IDs in evaluation scripts:

```text
s2lte0: Stage 2 learned top-k escape semantic stream
s2sth0: Stage 2 static entropy ablation stream
s3urg0: Stage 3 uniform residual grid stream
s3rae0: Stage 3 learned residual AE stream
```

This is another container-only change: payloads, decoded images, quality, and
roundtrip behavior are unchanged.

```text
historical compact-v2 d32 b5 r0.25 position Huffman, CRC32 + short ID:
  Kodak:
    run: 20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_kodak_eval_compacthdr_crc32_shortid
    wandb: wandb/offline-run-20260627_125830-wke3i8sf
    total_payload_bpp: 0.018972
    full_stream_bpp: 0.031301
    PSNR delta: +0.3957 dB
  DIV2K first 100:
    run: 20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_div2k100_eval_compacthdr_crc32_shortid
    wandb: wandb/offline-run-20260627_125941-0qgk0vg6
    total_payload_bpp: 0.018973
    full_stream_bpp: 0.031302
    PSNR delta: +0.4601 dB
  CLIC professional valid:
    run: 20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_clicval64_eval_compacthdr_crc32_shortid
    wandb: wandb/offline-run-20260627_130007-om6jmrfq
    total_payload_bpp: 0.017873
    full_stream_bpp: 0.030202
    PSNR delta: +0.7170 dB

historical compact-v2 low-rate d32 b4 r0.25 position Huffman, CRC32 + short ID:
  Kodak:
    run: 20260627_stage3_residual_poshuff_d32_b4_r025_4096calib_kodak_eval_compacthdr_crc32_shortid
    wandb: wandb/offline-run-20260627_125856-19zqaznl
    total_payload_bpp: 0.016205
    full_stream_bpp: 0.028534
    PSNR delta: +0.3705 dB
  DIV2K first 100:
    run: 20260627_stage3_residual_poshuff_d32_b4_r025_4096calib_div2k100_eval_compacthdr_crc32_shortid
    wandb: wandb/offline-run-20260627_130047-owfc51s0
    total_payload_bpp: 0.016262
    full_stream_bpp: 0.028591
    PSNR delta: +0.3055 dB
  CLIC professional valid:
    run: 20260627_stage3_residual_poshuff_d32_b4_r025_4096calib_clicval64_eval_compacthdr_crc32_shortid
    wandb: wandb/offline-run-20260627_130118-b90yfyj5
    total_payload_bpp: 0.015172
    full_stream_bpp: 0.027502
    PSNR delta: +0.5649 dB

roundtrip_failure_count: 0 for all six short-ID compact CRC32 runs
```

## Decision

Use `d32 b5 r0.25 hybrid position/semantic-position g4 Huffman` with
`smoothing_count=0` as the active Stage 3 quality bootstrap, and use
`d32 b4 r0.25 hybrid position/semantic-position g8 Huffman` with
`smoothing_count=0` as the low-rate anchor:

```text
semantic payload bpp is about 0.0107 on Kodak
active detail payload bpp is about 0.00815 on Kodak
active total semantic+detail payload bpp is about 0.01888 on Kodak
active compact-v3 CRC32 full stream bpp is about 0.02390 on Kodak
active quality improves by about +0.40 dB PSNR and +0.00576 MS-SSIM
low-rate anchor detail payload bpp is about 0.00540 on Kodak
low-rate anchor total semantic+detail payload bpp is about 0.01612 on Kodak
low-rate anchor compact-v3 CRC32 full stream bpp is about 0.02113 on Kodak
residual grid clipping is 0.0 for both active and low-rate anchor settings
```

Reason:

```text
The b5/r0.25 setting improves PSNR, L1, and MS-SSIM on Kodak, DIV2K, and CLIC
with exact semantic/detail roundtrip while staying below 0.02 total payload
bpp. Position Huffman preserves the same decoded residual grid as global
Huffman and gives a small but consistent actual-bpp reduction. Decision 009
then supersedes both the pure b4 low-rate anchor and the pure b5 quality anchor
with a hybrid position/semantic-position Huffman selector that transmits its
mode bit and improves actual bpp on Kodak, DIV2K, and CLIC without changing
reconstruction. The active b4 low-rate anchor uses g8 with smoothing=0; the
active b5 quality anchor uses g4 with smoothing=0 after g4/g8/smoothing
actual-byte comparisons.
The compact CoSERBitstream header replaces the earlier
JSON-header evaluation for active full-stream bpp. Compact v3 CRC32 is the
active compact container for low-bitrate crop experiments.
```

Rejected as active bootstrap:

```text
pure semantic-position residual Huffman d32 b4 r0.25 g4:
  improves actual compact-v3 full-stream bpp on DIV2K100 and CLIC, but regresses
  Kodak by +0.000015 bpp versus the d32/b4/r0.25 position-Huffman anchor. It is
  superseded by the hybrid selector in Decision 009.

d32 b5 r0.25 position Huffman:
  same reconstruction as the active b5 hybrid selector, but slightly higher
  actual compact-v3 CRC32 full-stream bpp on Kodak, DIV2K100, CLIC professional
  valid 41, and CLIC professional+mobile valid 64.

d32 b5 r0.25 hybrid position/semantic-position g8:
  slightly better than b5 g4 on DIV2K100 by 0.000010 bpp, but worse on Kodak,
  CLIC professional valid 41, and CLIC professional+mobile valid 64. The
  four-dataset mean favors g4 smoothing=1 by about 0.000002 bpp, and favors
  g4 smoothing=0 by a larger margin.

d32 b5 r0.25 hybrid position/semantic-position g4 smoothing=1:
  improves over b5 g8 on Kodak and CLIC, but is superseded by the same g4
  grouping with smoothing=0.

learned residual AE d32 c3 b5 r0.25, no-rate 2500step:
  functional semantic/detail bitstream and positive quality delta, but detail
  payload rises to 0.01465 bpp and total payload to 0.02537 bpp on Kodak while
  PSNR improves only +0.2073 dB. It is worse than the static d32 b5 r0.25
  point for the Core MVP low-bitrate range.

learned residual AE d32 c3 b5 r0.25, rate-proxy 0.03:
  detail payload drops to 0.00434 bpp and total payload to 0.01507 bpp on
  Kodak, but the detail stream collapses to about two symbols and improves only
  +0.0112 dB / +0.00007 MS-SSIM. It is a useful low-rate diagnostic, not an
  active codec.

d32 b4 r0.50:
  lower detail bpp but slightly hurts MS-SSIM on Kodak.

d32 b5 r0.25 global residual Huffman:
  same reconstruction as position Huffman, but slightly higher actual detail
  payload bpp on Kodak, DIV2K, and CLIC.

d32 b4 r0.25 global residual Huffman:
  same reconstruction as position Huffman, but slightly higher actual detail
  payload bpp on Kodak, DIV2K, and CLIC.

d32 b6 r0.25:
  slightly better quality than b5, but the extra payload is too large for the
  small gain in the Core MVP low-bitrate range.

d32 b4 r0.25 zlib_fixed_bits:
  positive quality/bpp tradeoff, but superseded by static residual Huffman.

d16 b3 r0.50:
  worse quality and no bitrate advantage after zlib.

d16 b4 r0.25:
  best quality among the quick sweep, but detail payload is too large for the
  low-bitrate Core MVP target unless used as a high-rate oracle.
```

## Next Actions

```text
1. Keep d32 b5 r0.25 hybrid position/semantic-position g4 Huffman with
   smoothing=0 as the active quality bootstrap and d32 b4 r0.25 hybrid
   position/semantic-position g8 Huffman with smoothing=0 as the low-rate
   anchor.
2. Rework the learned residual AE around an explicit entropy objective,
   teacher-distilled residual target, or spatially adaptive residual allocation
   before giving it longer training.
3. Replace fixed residual coding only after the learned detail path beats the
   hybrid-Huffman anchors under actual CoSERBitstream bytes.
4. Use the cross-dataset residual diagnostics: range 0.25 is not clipping on
   Kodak, DIV2K, or CLIC, so prioritize conditional entropy/detail allocation
   over range expansion.
5. Continue to evaluate only with actual CoSERBitstream bytes and exact
   semantic/detail roundtrip.
6. Use Decision 008 compact v3 CRC32 numbers for current full-stream bpp; the
   older compact-v2 short-ID section is retained as history.
7. Use Decision 009 for the semantic-conditioned residual Huffman follow-up:
   pure g4 is superseded, while the transmitted-mode hybrid selector is the
   active b4 low-rate and b5 quality anchor.
```
