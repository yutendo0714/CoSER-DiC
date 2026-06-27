# Decision 009: Stage 3 Semantic-Conditioned Residual Huffman

Date: 2026-06-27

## Context

Decision 007 established the fixed d32 residual grid as the Stage 3 bootstrap.
Decision 008 made compact-v3 CRC32 CoSERBitstream bytes the active full-stream
metric. The remaining static residual entropy gap was not from residual range:
d32/b4/r0.25 and d32/b5/r0.25 have near-zero clipping on Kodak, DIV2K, and
CLIC. The next plausible low-risk improvement was conditional entropy coding.

This probe keeps the decoded residual grid unchanged and changes only the
transmitted detail entropy model:

```text
semantic stream: learned top-k/escape Huffman token prior
detail stream: d32/b4/r0.25 residual grid
new condition: detail Huffman table is selected by semantic token group and
               detail grid position
side information: none per image; token-to-group mapping is stored in the prior
actual bpp: compact-v3 CoSERBitstream bytes with CRC32
```

The design intentionally stays simple. Semantic groups are obtained by k-means
over the Stage 1 VQ codebook embeddings, not by adding a learned side model.

## Implementation

```text
codec:
  src/coserdic/entropy/residual_grid.py
  StaticResidualGridSemanticPositionHuffmanCode
  StaticResidualGridHybridHuffmanCode

fit:
  scripts/fit_stage3_residual_huffman_prior.py
  --coding-mode semantic_position_huffman

eval:
  scripts/eval_stage3_uniform_residual_bitstream.py
  --detail-codec semantic_position_huffman
  --detail-codec hybrid_huffman

hybrid prior build:
  scripts/build_stage3_hybrid_residual_prior.py

tests:
  tests/test_residual_grid.py
```

The decoder reconstructs semantic tokens first, then uses those decoded tokens
as the context for detail residual entropy decoding. Evaluation checks both the
local detail-code roundtrip and the stream-level semantic/detail roundtrip. The
hybrid codec transmits one explicit mode bit per image, then decodes either the
position-Huffman payload or the semantic-position payload.

## Calibration Sweep

All priors were fitted on 8192 256x256 crops from OpenImages train plus DIV2K.

```text
g2:
  run: 20260627_stage3_residual_semposhuff_g2_d32_b4_r025_8192calib_openimages_div2k
  wandb: wandb/offline-run-20260627_134529-r4bxhauh
  mean_huffman_bits_per_symbol: 1.940376
  Kodak full_stream_bpp: 0.021296

g4:
  run: 20260627_stage3_residual_semposhuff_g4_d32_b4_r025_8192calib_openimages_div2k
  wandb: wandb/offline-run-20260627_134339-cdts131i
  mean_huffman_bits_per_symbol: 1.930738

g8:
  run: 20260627_stage3_residual_semposhuff_g8_d32_b4_r025_8192calib_openimages_div2k
  wandb: wandb/offline-run-20260627_133835-twtnrkgu
  mean_huffman_bits_per_symbol: 1.925484

g16:
  run: 20260627_stage3_residual_semposhuff_g16_d32_b4_r025_8192calib_openimages_div2k
  wandb: wandb/offline-run-20260627_134954-dwmo2nk4
  mean_huffman_bits_per_symbol: 1.926224
```

The g8 prior is the best calibration point, but g4 is more robust on Kodak.
The g2 prior underuses the semantic context. The g16 prior does not improve over
g8 and is more fragmented, so it was not evaluated further.

## Actual-Byte Results

All numbers are compact-v3 CRC32 CoSERBitstream full-stream bpp unless noted.
Roundtrip failure count was zero for every run below.

```text
Baseline low-rate d32/b4/r0.25 position Huffman:
  Kodak:  0.021205
  DIV2K100: 0.021266
  CLIC professional valid 41: 0.020171
  CLIC professional+mobile valid 64: 0.020405

Semantic-position g4:
  Kodak:
    run: 20260627_stage3_residual_semposhuff_g4_d32_b4_r025_8192calib_kodak_eval_compactv3_crc32
    wandb: wandb/offline-run-20260627_134406-ywxky0zy
    detail_payload_bpp: 0.005493
    total_payload_bpp: 0.016215
    full_stream_bpp: 0.021220
    delta vs position b4: +0.000015
    PSNR delta vs semantic-only: +0.3705 dB

  DIV2K first 100:
    run: 20260627_stage3_residual_semposhuff_g4_d32_b4_r025_8192calib_div2k100_eval_compactv3_crc32
    wandb: wandb/offline-run-20260627_134653-eao2bile
    detail_payload_bpp: 0.005690
    total_payload_bpp: 0.016115
    full_stream_bpp: 0.021119
    delta vs position b4: -0.000146
    PSNR delta vs semantic-only: +0.3055 dB

  CLIC professional valid 41:
    run: 20260627_stage3_residual_semposhuff_g4_d32_b4_r025_8192calib_clicpro41_eval_compactv3_crc32
    wandb: wandb/offline-run-20260627_134724-uk0ynpy0
    detail_payload_bpp: 0.005207
    total_payload_bpp: 0.015113
    full_stream_bpp: 0.020118
    delta vs position b4: -0.000054
    PSNR delta vs semantic-only: +0.5649 dB

  CLIC professional+mobile valid 64:
    run: 20260627_stage3_residual_semposhuff_g4_d32_b4_r025_8192calib_clicval64_eval_compactv3_crc32
    wandb: wandb/offline-run-20260627_134821-8xe59otf
    detail_payload_bpp: 0.005255
    total_payload_bpp: 0.015305
    full_stream_bpp: 0.020309
    delta vs position b4: -0.000096
    PSNR delta vs semantic-only: +0.5438 dB
```

The g8 prior gives the best DIV2K100 result but a larger Kodak regression:

```text
g8 Kodak full_stream_bpp: 0.021245
g8 DIV2K100 full_stream_bpp: 0.021101
g8 CLIC professional valid 41 full_stream_bpp: 0.020118
```

## Hybrid Selector Follow-Up

The g4 semantic-position prior improved cross-domain bpp but had a tiny Kodak
regression. A hybrid prior was first built by combining the 8192-crop b4
position prior and the g4 semantic-position prior:

```text
prior: outputs/stage3_residual_entropy/20260627_stage3_residual_hybrid_pos_semposg4_d32_b4_r025_8192calib_openimages_div2k/static_residual_grid_hybrid_huffman_prior.json
mode bit: 0 = position Huffman, 1 = semantic-position Huffman
selection: minimum actual payload byte length after adding the mode bit
```

Actual compact-v3 CRC32 results:

```text
Hybrid position/semantic-position g4:
  Kodak:
    run: 20260627_stage3_residual_hybrid_pos_semposg4_d32_b4_r025_8192calib_kodak_eval_compactv3_crc32
    wandb: wandb/offline-run-20260627_135730-27tc4yrq
    detail_payload_bpp: 0.005432
    total_payload_bpp: 0.016154
    full_stream_bpp: 0.021159
    delta vs position b4: -0.000046
    semantic-position mode rate: 0.5000

  DIV2K first 100:
    run: 20260627_stage3_residual_hybrid_pos_semposg4_d32_b4_r025_8192calib_div2k100_eval_compactv3_crc32
    wandb: wandb/offline-run-20260627_135822-dwqeioda
    detail_payload_bpp: 0.005649
    total_payload_bpp: 0.016074
    full_stream_bpp: 0.021079
    delta vs position b4: -0.000187
    semantic-position mode rate: 0.6500

  CLIC professional valid 41:
    run: 20260627_stage3_residual_hybrid_pos_semposg4_d32_b4_r025_8192calib_clicpro41_eval_compactv3_crc32
    wandb: wandb/offline-run-20260627_135857-e8675327
    detail_payload_bpp: 0.005106
    total_payload_bpp: 0.015012
    full_stream_bpp: 0.020017
    delta vs position b4: -0.000155
    semantic-position mode rate: 0.6341

  CLIC professional+mobile valid 64:
    run: 20260627_stage3_residual_hybrid_pos_semposg4_d32_b4_r025_8192calib_clicval64_eval_compactv3_crc32
    wandb: wandb/offline-run-20260627_135936-m93cc9h7
    detail_payload_bpp: 0.005178
    total_payload_bpp: 0.015228
    full_stream_bpp: 0.020233
    delta vs position b4: -0.000172
    semantic-position mode rate: 0.6875
```

The hybrid selector preserves the same decoded b4 residual grid, has zero
roundtrip failures, and improves actual full-stream bpp on every checked
dataset. The mode bit is explicit decoder-visible side information, so the
comparison is fair under transmitted bytes.

The same selector was then evaluated with the g8 semantic-position prior:

```text
Hybrid position/semantic-position g8:
  prior: outputs/stage3_residual_entropy/20260627_stage3_residual_hybrid_pos_semposg8_d32_b4_r025_8192calib_openimages_div2k/static_residual_grid_hybrid_huffman_prior.json

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

g8 hybrid is better than g4 hybrid on Kodak, DIV2K100, and CLIC professional
valid 41. It is worse on CLIC professional+mobile valid 64 by only 0.000002 bpp.

The active b4 g8 prior was then refit with `smoothing_count=0`, which improved
all four checked actual-byte results while preserving exact roundtrip:

```text
Hybrid position/semantic-position g8 smoothing=0, d32/b4/r0.25:
  prior: outputs/stage3_residual_entropy/20260627_stage3_residual_hybrid_pos_semposg8_sm0_d32_b4_r025_8192calib_openimages_div2k/static_residual_grid_hybrid_huffman_prior.json
  Kodak full_stream_bpp: 0.021128
  DIV2K100 full_stream_bpp: 0.021039
  CLIC professional valid 41 full_stream_bpp: 0.019990
  CLIC professional+mobile valid 64 full_stream_bpp: 0.020226
```

## Quality Bootstrap Group-Count Follow-Up

The same hybrid selector was then applied to the d32/b5/r0.25 quality
bootstrap. This keeps the decoded b5 residual grid unchanged and changes only
the transmitted detail entropy payload. The first check used g8, then g4 was
fitted and evaluated because the active decision should not assume that the
best b4 group count is also best for b5. Finally, g4 was refit with
`smoothing_count=0` to test whether the default Laplace smoothing was too
conservative for the 8192-crop calibration set.

```text
Hybrid position/semantic-position g4 smoothing=0, d32/b5/r0.25:
  prior: outputs/stage3_residual_entropy/20260627_stage3_residual_hybrid_pos_semposg4_sm0_d32_b5_r025_8192calib_openimages_div2k/static_residual_grid_hybrid_huffman_prior.json

  Kodak:
    run: 20260627_stage3_residual_hybrid_pos_semposg4_sm0_d32_b5_r025_8192calib_kodak_eval_compactv3_crc32
    wandb: wandb/offline-run-20260627_143205-fdb9h7of
    detail_payload_bpp: 0.008153
    total_payload_bpp: 0.018875
    full_stream_bpp: 0.023880
    previous g4 smoothing=1 full_stream_bpp: 0.023885
    previous g8 hybrid full_stream_bpp: 0.023895
    previous position b5 full_stream_bpp: 0.023972
    PSNR delta vs semantic-only: +0.3957 dB
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
    PSNR delta vs semantic-only: +0.4601 dB
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
    PSNR delta vs semantic-only: +0.7171 dB
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
    PSNR delta vs semantic-only: +0.6829 dB
    semantic-position mode rate: 0.7656
```

All b5 g4/g8 and smoothing evaluations have exact semantic/detail roundtrip and
`roundtrip_failure_count: 0`. g4 smoothing=0 wins Kodak, CLIC professional
valid 41, and CLIC professional+mobile valid 64. g8 remains ahead on DIV2K100
by only 0.000001 bpp. The four-dataset mean favors g4 smoothing=0, so it is the
active b5 quality prior.

## Decision

Promote the hybrid position/semantic-position g8 Huffman prior with
smoothing=0 as the active d32/b4/r0.25 low-rate anchor. Promote the hybrid
position/semantic-position g4 Huffman prior with smoothing=0 as the active
d32/b5/r0.25 quality bootstrap.
The previous d32/b5/r0.25 position-Huffman, g4 smoothing=1, and g8-hybrid
priors remain clean ablations, but are no longer the active quality setting.

Reason:

```text
1. The active hybrid priors improve actual compact-v3 CRC32 full-stream bpp on
   Kodak, DIV2K100, CLIC professional valid, and CLIC professional+mobile valid
   for both b4 and b5 residual grids compared with pure position Huffman.
2. It changes only entropy coding, not the decoded residual grid, so the quality
   deltas are preserved.
3. The transmitted mode bit makes the per-image selection explicit and auditable.
4. The result supports the CoSER-DiC semantic/detail separation without adding a
   learned side model.
```

## Next

```text
1. Consider smoothing-count sweeps for the active g4/g8 priors only if the next
   learned residual entropy step does not dominate this static anchor.
2. Keep all promotion decisions tied to actual compact-v3 CoSERBitstream bytes,
   not estimated Huffman bits.
3. Preserve the previous b5 and b4 position-Huffman priors plus the b5 g4
   smoothing=1 and g8 hybrid priors as ablation anchors.
4. Use the hybrid mode-rate metric as a diagnostic for future learned
   semantic-conditioned entropy models.
```
