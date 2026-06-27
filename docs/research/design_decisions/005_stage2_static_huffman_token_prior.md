# Stage 2 Static Semantic Token Priors

Date: 2026-06-27  
Status: Static Stage 2 bootstrap baseline; superseded as active path by learned top-k/escape  
Parent policy: `docs/research/design_decisions/003_official_implementation_reference_policy.md`

## Purpose

Stage 2 needs to prove that the Stage 1 semantic token stream is actually
compressible beyond fixed-width token packing, using bytes emitted by a real
encoder/decoder path.

The first Stage 2 implementation is therefore a simple CoSER-owned static
semantic-token coder:

```text
Stage 1 semantic VQ indices
  -> decoder-known static categorical/context code
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
optional position-conditioned static Huffman code
optional left-context static Huffman code
optional static categorical ANS code through CompressAI
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

Static coding is intentionally simpler:
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

## Position-Conditioned Static Huffman Ablation

Motivation:

```text
The semantic token grid is spatially structured. A natural next step is to give
each token position a decoder-known Huffman code, approximating a simple
position-conditioned prior without transmitting side information.
```

Implemented:

```text
StaticPositionHuffmanCode
--prior-kind position
--position-backoff-mass
```

Results:

```text
20260627_stage2_position_huffman_prior_512calib_from_stage1_best
  smoothing_count: 1
  calibration payload bpp: 0.012668

20260627_stage2_position_huffman_smooth1_512calib_kodak_eval
  Kodak payload bpp: 0.012700
  fixed_bits payload bpp: 0.012695
  delta vs fixed_bits: +0.000005

20260627_stage2_position_huffman_prior_512calib_smooth0_from_stage1_best
  smoothing_count: 0
  calibration payload bpp: 0.007783

20260627_stage2_position_huffman_smooth0_512calib_kodak_eval
  Kodak payload bpp: 0.019918
  fixed_bits payload bpp: 0.012695
  delta vs fixed_bits: +0.007222

20260627_stage2_position_huffman_prior_512calib_backoff256_from_stage1_best
  smoothing_count: 0
  position_backoff_mass: 256
  calibration payload bpp: 0.008726

20260627_stage2_position_huffman_backoff256_512calib_kodak_eval
  Kodak payload bpp: 0.013514
  fixed_bits payload bpp: 0.012695
  delta vs fixed_bits: +0.000819
```

Interpretation:

```text
Position-conditioned static Huffman is not adopted as the active Stage 2 prior.

With smoothing_count=1, each position has too little data relative to the
8192-code vocabulary, so the pseudo-counts make the code nearly uniform.

With smoothing_count=0, calibration payload bpp looks excellent, but Kodak
generalization fails because low-frequency/unseen position-token pairs receive
very long codes.

Global-distribution backoff reduces the overfitting but still does not beat the
global static Huffman prior on Kodak.
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

## 4096-Image Calibration Update

The original 512-calib runs actually used only 256 images because the Stage 1
checkpoint config pointed to:

```text
/dpl/race_pilot_openimages_crops
/dpl/race_pilot_div2k_crops_v3
```

Those folders contain 192 + 64 images. A larger decoder-known prior can be fit
from available datasets without changing the codec:

```text
/dpl/open-images-v6/train/data
/dpl/div2k
```

### Global Static Huffman

```text
20260627_stage2_static_huffman_prior_4096calib_oi_div2k_from_stage1_best

wandb:
  wandb/offline-run-20260627_100002-dkj3ec9s

prior:
  outputs/stage2_semantic_entropy/20260627_stage2_static_huffman_prior_4096calib_oi_div2k_from_stage1_best/static_huffman_prior.json
```

Calibration:

```text
num_images: 4096
total_tokens: 262144
active_codes: 7520 / 8192
global_entropy_bits: 11.9197 bits/token
mean_huffman_bits_per_token: 11.9561
mean_huffman_payload_bpp: 0.011729
fixed_bits_payload_bpp: 0.012695
```

Compact-header Kodak evaluation:

```text
20260627_stage2_static_huffman_4096calib_oi_div2k_kodak_eval_compacthdr

wandb:
  wandb/offline-run-20260627_100216-ktw8unui

payload_bpp: 0.011820
full_stream_bpp: 0.071533
payload_bytes/image: 96.83
stream_bytes/image: 586.00
fixed_bits payload_bpp: 0.012695
fixed_bits full_stream_bpp: 0.072510
delta payload vs fixed_bits: -0.000875 bpp
token_roundtrip: true
PSNR / L1 / MS-SSIM: unchanged from Stage 1 semantic reconstruction
```

### Left-Context Static Huffman

Implemented:

```text
StaticLeftContextHuffmanCode
--prior-kind left_context
--context-topk
--context-backoff-mass
```

Decoder-known context:

```text
context at (y, x) =
  BOS if x == 0
  exact context bucket if decoded left token is one of top-k frequent tokens
  OTHER otherwise
```

This preserves a real causal decoding boundary: no encoder-side token
probabilities or per-image side information are transmitted.

256-image calibration overfit badly. Kodak payload bpp was worse than fixed
bits for top0/top8/top16/top32/top64 sweeps, even when calibration bpp looked
excellent. With 4096 calibration images and strong global backoff, it
generalizes:

```text
20260627_stage2_leftctx_huffman_top64_b4096_4096calib_oi_div2k_from_stage1_best

wandb:
  wandb/offline-run-20260627_100057-9ry5exbw

prior:
  outputs/stage2_semantic_entropy/20260627_stage2_leftctx_huffman_top64_b4096_4096calib_oi_div2k_from_stage1_best/static_left_context_huffman_prior.json
```

Calibration:

```text
num_images: 4096
context_topk: 64
context_backoff_mass: 4096
active_codes: 7520 / 8192
mean_huffman_bits_per_token: 11.7991
mean_huffman_payload_bpp: 0.011575
fixed_bits_payload_bpp: 0.012695
```

Compact-header Kodak evaluation:

```text
20260627_stage2_leftctx_huffman_top64_b4096_4096calib_oi_div2k_kodak_eval_compacthdr

wandb:
  wandb/offline-run-20260627_100231-92tm3m06

payload_bpp: 0.011729
full_stream_bpp: 0.071437
payload_bytes/image: 96.08
stream_bytes/image: 585.21
fixed_bits payload_bpp: 0.012695
fixed_bits full_stream_bpp: 0.072510
delta payload vs fixed_bits: -0.000966 bpp
token_roundtrip: true
PSNR / L1 / MS-SSIM: unchanged from Stage 1 semantic reconstruction
```

### Static ANS Probe

Implemented:

```text
StaticANSCode
--prior-kind ans
CompressAI BufferedRansEncoder / RansDecoder
```

Result:

```text
20260627_stage2_static_ans_prior_4096calib_oi_div2k_from_stage1_best
  wandb/offline-run-20260627_100550-vr54e2zr

20260627_stage2_static_ans_4096calib_oi_div2k_kodak_eval_compacthdr
  wandb/offline-run-20260627_100606-1w8hj69p

Kodak payload_bpp: 0.012451
Kodak full_stream_bpp: 0.072255
delta payload vs fixed_bits: -0.000244 bpp
token_roundtrip: true
```

Interpretation:

```text
Static ANS is valid and useful as future infrastructure, but it is not the
active coder for 64-token semantic streams. Per-image ANS flush overhead makes
it worse than Huffman here.
```

## Static Baseline Decision

Best static Stage 2 semantic-token baseline:

```text
Use 4096-calibrated left-context static Huffman:

  outputs/stage2_semantic_entropy/20260627_stage2_leftctx_huffman_top64_b4096_4096calib_oi_div2k_from_stage1_best/static_left_context_huffman_prior.json

Kodak payload_bpp: 0.011729
Kodak full_stream_bpp: 0.071437
token_roundtrip: true
```

Fallback baseline:

```text
Use 4096-calibrated global static Huffman when a context-free baseline is
needed.
```

Rejected as active baselines:

```text
256-image position-conditioned static Huffman
256-image left-context static Huffman
4096-image static ANS
```

Supersession:

```text
The 32768-token learned top512/escape Huffman bridge later beat this static
baseline with actual Kodak payload_bpp 0.010722 and exact token roundtrip.
The static left-context prior remains the main static ablation and safety
baseline, not the current active research path.
```

## Next Step

Continue from the CoSER-owned neural categorical token prior:

```text
input:
  semantic token grid and/or decoder-known positional features

coding target:
  decoder-rebuilt neural logits with RDVQ-style top-k/escape coding
  learned top512/escape Huffman as the current actual-byte bridge
  use StaticANSCode only for decoder-known index/CDF probes unless a chunked
  causal decoder is added
```

The first learned-prior probe and its adoption boundary are documented in:

```text
docs/research/design_decisions/006_stage2_learned_token_prior_probe.md
```
