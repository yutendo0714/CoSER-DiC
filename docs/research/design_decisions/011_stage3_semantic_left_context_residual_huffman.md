# Decision 011: Stage 3 Semantic-Position-Left Residual Huffman

Date: 2026-06-27

## Context

Decision 009 introduced semantic-position residual Huffman coding for the fixed
d32 residual grid. The best hybrid selector improved actual payload bpp, but it
still treated residual symbols at each detail position as conditionally
independent after semantic group and position selection.

The next low-risk entropy improvement is a causal detail context that the
decoder can reproduce while decoding the detail stream. This preserves the Core
MVP split:

```text
semantic stream: learned top-k/escape Huffman token prior
detail stream: d32 residual-grid entropy payload
new detail context: semantic token group + detail position + left residual sign
per-image side information: none
paper metric: actual_payload_bpp
debug metric: debug_full_stream_bpp
```

The left context is deliberately simple:

```text
0: row start
1: left residual code below the quantizer zero code
2: left residual code equals the quantizer zero code
3: left residual code above the quantizer zero code
```

The detail stream is encoded and decoded in channel-major raster order, so the
left residual code is already decoded whenever it is needed. No per-image mask,
rate map, table, or prompt is transmitted.

## Implementation

```text
codec:
  src/coserdic/entropy/residual_grid.py
  StaticResidualGridSemanticPositionLeftContextHuffmanCode

fit:
  scripts/fit_stage3_residual_huffman_prior.py
  --coding-mode semantic_position_leftctx_huffman

eval:
  scripts/eval_stage3_uniform_residual_bitstream.py
  --detail-codec semantic_position_leftctx_huffman

tests:
  tests/test_residual_grid.py
```

The prior stores fixed decoder-known Huffman code lengths for every
`channel, detail_y, detail_x, semantic_group, left_context` tuple. Empty
calibration contexts use a uniform fixed table when smoothing is zero, avoiding
invalid zero-count Huffman tables without adding image-specific bits.

## Calibration

Both active priors were fit on 8192 256x256 crops from OpenImages train plus
DIV2K, using the active Stage 1 semantic VQ checkpoint and the active Stage 2
learned top-k/escape semantic prior.

```text
b4 low-rate prior:
  run: 20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_openimages_div2k
  wandb: wandb/offline-run-20260627_153936-20g6oib6
  semantic groups: 4
  smoothing_count: 0
  mean_huffman_bits_per_symbol: 1.788819
  previous semantic-position-left g8 smoothing0: 1.779746
  previous semantic-position g8 smoothing0: 1.917438

b5 quality prior:
  run: 20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_8192calib_openimages_div2k
  wandb: wandb/offline-run-20260627_152106-v74n0yot
  semantic groups: 4
  smoothing_count: 0
  mean_huffman_bits_per_symbol: 2.697383
  previous semantic-position g4 smoothing0: 2.855390
```

## Actual Payload Results

All values below use compact-v3 CRC32 stream auditing, but the main paper
comparison metric is `actual_payload_bpp`, which counts only entropy-coded
semantic and detail payload streams. Roundtrip failure count is zero for every
run.

```text
b4 low-rate, d32/b4/r0.25, semantic-position-left g4:
  Kodak:
    run: 20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_kodak_eval_compactv3_crc32
    wandb: wandb/offline-run-20260627_154002-l0fvhzxf
    semantic_payload_bpp: 0.010722
    detail_payload_bpp: 0.005107
    actual_payload_bpp: 0.015828
    debug_full_stream_bpp: 0.020833
    PSNR delta: +0.3705 dB

  DIV2K first 100:
    run: 20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_div2k100_eval_compactv3_crc32
    wandb: wandb/offline-run-20260627_154056-oufto3f2
    semantic_payload_bpp: 0.010425
    detail_payload_bpp: 0.005233
    actual_payload_bpp: 0.015658
    debug_full_stream_bpp: 0.020663
    PSNR delta: +0.3055 dB

  CLIC professional valid 41:
    run: 20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_clicpro41_eval_compactv3_crc32
    wandb: wandb/offline-run-20260627_154129-4p6l28qx
    semantic_payload_bpp: 0.009906
    detail_payload_bpp: 0.004802
    actual_payload_bpp: 0.014708
    debug_full_stream_bpp: 0.019713
    PSNR delta: +0.5649 dB

  CLIC professional+mobile valid 64:
    run: 20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_clicval64_eval_compactv3_crc32
    wandb: wandb/offline-run-20260627_154209-kxhgrqat
    semantic_payload_bpp: 0.010050
    detail_payload_bpp: 0.004881
    actual_payload_bpp: 0.014931
    debug_full_stream_bpp: 0.019936
    PSNR delta: +0.5438 dB

b5 quality, d32/b5/r0.25, semantic-position-left g4:
  Kodak:
    run: 20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_8192calib_kodak_eval_compactv3_crc32
    wandb: wandb/offline-run-20260627_152133-92lwm3b4
    semantic_payload_bpp: 0.010722
    detail_payload_bpp: 0.007751
    actual_payload_bpp: 0.018473
    debug_full_stream_bpp: 0.023478
    PSNR delta: +0.3957 dB

  DIV2K first 100:
    run: 20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_8192calib_div2k100_eval_compactv3_crc32
    wandb: wandb/offline-run-20260627_152426-7u7pb0ua
    semantic_payload_bpp: 0.010425
    detail_payload_bpp: 0.007965
    actual_payload_bpp: 0.018390
    debug_full_stream_bpp: 0.023395
    PSNR delta: +0.4601 dB

  CLIC professional valid 41:
    run: 20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_8192calib_clicpro41_eval_compactv3_crc32
    wandb: wandb/offline-run-20260627_152456-qd8fqs38
    semantic_payload_bpp: 0.009906
    detail_payload_bpp: 0.007351
    actual_payload_bpp: 0.017257
    debug_full_stream_bpp: 0.022261
    PSNR delta: +0.7171 dB

  CLIC professional+mobile valid 64:
    run: 20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_8192calib_clicval64_eval_compactv3_crc32
    wandb: wandb/offline-run-20260627_152534-dpm1cmqs
    semantic_payload_bpp: 0.010050
    detail_payload_bpp: 0.007437
    actual_payload_bpp: 0.017487
    debug_full_stream_bpp: 0.022491
    PSNR delta: +0.6829 dB
```

## Comparison To Previous Active Anchors

```text
b4 previous active hybrid g8 smoothing0 -> semantic-position-left g4:
  Kodak:   0.016123 -> 0.015828  (-0.000295 actual_payload_bpp)
  DIV2K100: 0.016034 -> 0.015658 (-0.000376)
  CLIC41:  0.014985 -> 0.014708  (-0.000277)
  CLIC64:  0.015221 -> 0.014931  (-0.000290)

b5 previous active hybrid g4 smoothing0 -> semantic-position-left g4:
  Kodak:   0.018875 -> 0.018473  (-0.000402 actual_payload_bpp)
  DIV2K100: 0.018853 -> 0.018390 (-0.000463)
  CLIC41:  0.017727 -> 0.017257  (-0.000470)
  CLIC64:  0.017931 -> 0.017487  (-0.000444)
```

## Decision

Promote `semantic_position_leftctx_huffman` as the active Stage 3 residual
payload codec for the Core MVP bootstrap:

```text
quality anchor:
  d32/b5/r0.25, semantic groups 4, smoothing 0

low-rate anchor:
  d32/b4/r0.25, semantic groups 4, smoothing 0
```

This is preferred over the previous hybrid selector because it lowers actual
payload bpp on all checked datasets without adding an explicit per-image mode
bit and without changing reconstruction quality.

## Group-Count Checks

Additional same-day sweeps checked whether the active group counts should be
changed.

```text
b5 g8 diagnostic:
  fit run: 20260627_stage3_residual_semposleft_g8_sm0_d32_b5_r025_8192calib_openimages_div2k
  wandb: wandb/offline-run-20260627_153522-dc1y9y7q
  mean_huffman_bits_per_symbol: 2.672975
  actual_payload_bpp:
    Kodak: 0.018489
    DIV2K100: 0.018413
    CLIC41: 0.017257
    CLIC64: 0.017504
  decision: reject for active b5; g4 has lower four-dataset actual payload mean.

b4 g8 diagnostic:
  fit run: 20260627_stage3_residual_semposleft_g8_sm0_d32_b4_r025_8192calib_openimages_div2k
  wandb: wandb/offline-run-20260627_151532-r1g36eh3
  mean_huffman_bits_per_symbol: 1.779746
  actual_payload_bpp:
    Kodak: 0.015828
    DIV2K100: 0.015671
    CLIC41: 0.014702
    CLIC64: 0.014927
  decision: keep as CLIC-best candidate, but use simpler g4 as active low-rate
            because its four-dataset actual payload mean is marginally lower.
```

Next checks:

1. Add a per-image comparison report to detect rare cases where g8 or hybrid
   remains better.
2. Keep `actual_payload_bpp` as the main paper bpp and
   `debug_full_stream_bpp` as the roundtrip audit metric.
