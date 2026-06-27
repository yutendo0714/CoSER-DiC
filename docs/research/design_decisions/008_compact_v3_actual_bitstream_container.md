# Decision 008: Compact v3 Actual-Byte Container

Date: 2026-06-27

## Context

Core MVP evaluation must use actual bytes produced by `compress/decompress`,
not estimated bitrate. After moving from JSON headers to compact headers, fixed
container overhead was still visible at 256x256 low-bitrate crop scale:

```text
Stage 2 Kodak semantic-only, compact-v2 CRC32 short ID:
  payload_bpp: 0.010721842447916666
  full_stream_bpp: 0.021830240885416668

Stage 3 Kodak d32/b5/r0.25 position Huffman, compact-v2 CRC32 short ID:
  total_payload_bpp: 0.018971761067708332
  full_stream_bpp: 0.031300862630208336
```

This overhead was a container artifact, not a codec property.

## Decision

Promote compact v3 as the active `CoSERBitstream(header_codec="compact",
checksum_codec="crc32")` encoding for new Core MVP bitstreams.

Compact v3 keeps the same high-level header fields but encodes them with:

```text
varint image dimensions, shapes, and payload lengths
zigzag varint signed rate/perception IDs
CRC32 checksum for crop-level corruption detection
small table IDs for common codec_version, color_space, and entropy_model_version strings
fallback literal strings for unknown future IDs
```

Old compact v1/v2 decoding remains supported. New streams use compact v3.

## Actual-Byte Results

Stage 2 semantic-only learned top512/escape:

```text
Kodak 24:
  run: 20260627_stage2_learned_topk512_escape_huffman_32768tokens_8kprior_kodak_eval_decodersched_compactv3_crc32_shortid
  wandb: wandb/offline-run-20260627_130859-k3q9zxxr
  payload_bpp: 0.010721842447916666
  full_stream_bpp: 0.015482584635416666

DIV2K first 100:
  run: 20260627_stage2_learned_topk512_escape_huffman_32768tokens_8kprior_div2k100_eval_decodersched_compactv3_crc32_shortid
  wandb: wandb/offline-run-20260627_130938-fro0s4q0
  payload_bpp: 0.0104248046875
  full_stream_bpp: 0.015185546875

CLIC professional valid, 41 usable crops:
  run: 20260627_stage2_learned_topk512_escape_huffman_32768tokens_8kprior_clicproval64_eval_decodersched_compactv3_crc32_shortid
  wandb: wandb/offline-run-20260627_131919-r02hxt9a
  payload_bpp: 0.009905559260670731
  full_stream_bpp: 0.014666301448170731

CLIC professional+mobile valid, first 64:
  run: 20260627_stage2_learned_topk512_escape_huffman_32768tokens_8kprior_clicval64_eval_decodersched_compactv3_crc32_shortid
  wandb: wandb/offline-run-20260627_131015-kqco3khp
  payload_bpp: 0.010049819946289062
  full_stream_bpp: 0.014810562133789062
```

Stage 3 active d32/b5/r0.25 residual position Huffman:

```text
Kodak 24:
  run: 20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_kodak_eval_compactv3_crc32_shortid
  wandb: wandb/offline-run-20260627_131040-5rsh6r1j
  total_payload_bpp: 0.018971761067708332
  full_stream_bpp: 0.023976643880208332
  PSNR delta vs semantic-only: +0.3957391579945906 dB

DIV2K first 100:
  run: 20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_div2k100_eval_compactv3_crc32_shortid
  wandb: wandb/offline-run-20260627_131119-i941bt45
  total_payload_bpp: 0.018973388671875
  full_stream_bpp: 0.0239794921875
  PSNR delta vs semantic-only: +0.4601301479339597 dB

CLIC professional valid, 41 usable crops:
  run: 20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_clicproval64_eval_compactv3_crc32_shortid
  wandb: wandb/offline-run-20260627_131947-7dho224l
  total_payload_bpp: 0.017872880144817072
  full_stream_bpp: 0.022877762957317072
  PSNR delta vs semantic-only: +0.7170708819133509 dB

CLIC professional+mobile valid, first 64:
  run: 20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_clicval64_eval_compactv3_crc32_shortid
  wandb: wandb/offline-run-20260627_131151-pbzv3rnm
  total_payload_bpp: 0.018075942993164062
  full_stream_bpp: 0.023080825805664062
  PSNR delta vs semantic-only: +0.6828677207231522 dB
```

Stage 3 low-rate d32/b4/r0.25 residual position Huffman:

```text
Kodak 24:
  run: 20260627_stage3_residual_poshuff_d32_b4_r025_4096calib_kodak_eval_compactv3_crc32_shortid
  wandb: wandb/offline-run-20260627_131212-mk0oi63a
  total_payload_bpp: 0.016204833984375
  full_stream_bpp: 0.021209716796875
  PSNR delta vs semantic-only: +0.37052687009175855 dB

DIV2K first 100:
  run: 20260627_stage3_residual_poshuff_d32_b4_r025_4096calib_div2k100_eval_compactv3_crc32_shortid
  wandb: wandb/offline-run-20260627_131251-pfi3wdwy
  total_payload_bpp: 0.01626220703125
  full_stream_bpp: 0.02126708984375
  PSNR delta vs semantic-only: +0.30548624038696204 dB

CLIC professional valid, 41 usable crops:
  run: 20260627_stage3_residual_poshuff_d32_b4_r025_4096calib_clicproval64_eval_compactv3_crc32_shortid
  wandb: wandb/offline-run-20260627_132013-sutp1qr1
  total_payload_bpp: 0.015172446646341464
  full_stream_bpp: 0.020177329458841462
  PSNR delta vs semantic-only: +0.5648934899306859 dB

CLIC professional+mobile valid, first 64:
  run: 20260627_stage3_residual_poshuff_d32_b4_r025_4096calib_clicval64_eval_compactv3_crc32_shortid
  wandb: wandb/offline-run-20260627_131325-2nf8ndjf
  total_payload_bpp: 0.015399932861328125
  full_stream_bpp: 0.020404815673828125
  PSNR delta vs semantic-only: +0.5438231080770493 dB
```

All compact v3 runs above have `roundtrip_failure_count: 0`.

## Interpretation

Compact v3 does not change the proposed CoSER-DiC modeling axis. It removes a
measurement artifact so the reported full-stream bpp is closer to the actual
semantic/detail payload cost. The effect is especially important at 256x256
low-bitrate crop scale; larger images will amortize the header naturally.

The active Core MVP anchors after this decision are:

```text
semantic-only Kodak full stream: 0.015483 bpp
Stage 3 b5 Kodak full stream: 0.023972 bpp, +0.3957 dB PSNR
Stage 3 b4 Kodak full stream: 0.021205 bpp, +0.3705 dB PSNR
```

## Follow-up

```text
1. Keep compact v3 for all new actual-byte evaluations.
2. Do not compare new compact-v3 full-stream bpp directly against old JSON or
   compact-v2 results without labeling the container format.
3. Keep reporting payload bpp beside full-stream bpp so model/payload
   improvements are separable from container overhead improvements.
4. If future work stores large side information, move the side information into
   entropy-coded payload sections instead of expanding the fixed header.
```
