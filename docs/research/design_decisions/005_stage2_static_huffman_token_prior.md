# Stage 2 Static Huffman Semantic Token Prior

Date: 2026-06-27  
Status: Active Stage 2 bootstrap decision  
Parent policy: `docs/research/design_decisions/003_official_implementation_reference_policy.md`

## Purpose

Stage 2 needs to prove that the Stage 1 semantic token stream is actually
compressible beyond fixed-width token packing, using bytes emitted by a real
encoder/decoder path.

The first Stage 2 implementation is therefore a simple CoSER-owned static
Huffman semantic-token coder:

```text
Stage 1 semantic VQ indices
  -> decoder-known static categorical code
  -> prefix-coded token payload bytes
  -> CoSERBitstream container
  -> exact token roundtrip
  -> semantic-only reconstruction
```

This is not the final entropy model. It is the bridge from deterministic
fixed_bits auditing to learned/contextual entropy coding.

## Implementation

```text
src/coserdic/entropy/static_huffman.py
scripts/fit_stage2_static_huffman_prior.py
scripts/eval_stage2_static_huffman_bitstream.py
tests/test_static_huffman.py
```

Design choices:

```text
canonical static Huffman code
code lengths are decoder-known model state
no per-image codebook side information
smoothing_count=1 so unseen symbols stay decodable
actual payload bytes are measured after bit packing
outer CoSERBitstream bytes are reported separately
```

Reference relation:

```text
RDVQ top-k/escape tensor-rANS remains the target L2 implementation reference
for the practical learned entropy-coded version.

Static Huffman is intentionally simpler:
  - no causal/context prior
  - no transmitted adaptive CDF
  - no neural token predictor
  - no arithmetic/rANS backend yet
```

## Calibration Run

```text
20260627_stage2_static_huffman_prior_512calib_from_stage1_best

checkpoint:
  checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt

wandb:
  wandb/offline-run-20260627_092913-cb1lzc77

prior:
  outputs/stage2_semantic_entropy/20260627_stage2_static_huffman_prior_512calib_from_stage1_best/static_huffman_prior.json
```

Calibration summary:

```text
num_images: 256
total_tokens: 16384
active_codes: 4725 / 8192
global_entropy_bits: 11.6919 bits/token
fixed_bits_per_token: 13
mean_huffman_bits_per_token: 11.9968
fixed_bits_payload_bpp: 0.012695
mean_huffman_payload_bpp: 0.011768
payload_bpp_delta_vs_fixed: -0.000927
```

## Kodak Evaluation

```text
20260627_stage2_static_huffman_512calib_kodak_eval

wandb:
  wandb/offline-run-20260627_092928-568umx2m
```

Actual byte-stream results on 24 Kodak center crops:

```text
static_huffman:
  payload_bpp: 0.012054
  full_stream_bpp: 0.074478
  payload_bytes/image: 98.75
  stream_bytes/image: 610.125
  token_roundtrip: true
  PSNR: 20.7363 dB
  L1: 0.06508
  MS-SSIM: 0.67577

fixed_bits in the same Stage 2 container:
  payload_bpp: 0.012695
  full_stream_bpp: 0.074707
  payload_bytes/image: 104.0
  stream_bytes/image: 612.0
  token_roundtrip: true
```

## Interpretation

Static Huffman improves actual semantic-token payload bpp on Kodak:

```text
0.012695 -> 0.012054 bpp
delta: -0.000641 bpp
relative payload reduction: about 5.0%
```

This is a modest but important positive result:

```text
1. The Stage 1 semantic tokens have exploitable non-uniformity.
2. Actual byte-stream coding can beat fixed_bits without changing quality.
3. Generic zlib was not enough on 64-token payloads, but token-aware coding is.
4. The next Stage 2 target should be learned/contextual token coding, not more
   generic byte compression.
```

The outer JSON CoSERBitstream container still dominates full bpp for this tiny
semantic-only stream. Keep reporting payload and full-stream bpp separately
until the production container is compacted.

## Next Step

Implement a CoSER-owned neural categorical token prior:

```text
input:
  semantic token grid and/or decoder-known positional features

starting prior:
  context-free learned logits or grouped row/column context

coding target:
  static Huffman baseline first
  then RDVQ-style top-k/escape arithmetic/rANS reference
```

