# 056 Stage 5 Component Codebook Huffman Control

Date: 2026-06-29

## Decision

Add static Huffman entropy coding for component-codebook basis-control symbols.

This is a rate-efficiency improvement for the Stage 5 counted condition-control
stream. It does not change the CoSER semantic/detail payload semantics and does
not add RGB postprocessing.

## Rationale

Component codebooks improve quantization quality at a fixed number of bits, but
their symbol distributions are still non-uniform and position-dependent.
Entropy-coding those level indices can reduce `actual_payload_bpp` for the same
decoded condition-control coefficients.

This is directly relevant to BD-rate: at the same Stage 5 decoded quality, a
lower actual payload bpp improves the LPIPS/DISTS rate curve.

## Bitstream Rule

The component codebooks and Huffman tables are fixed decoder-side state fitted
from non-eval train-cache coefficients.

Per image, only entropy-coded symbol streams are transmitted and counted:

```text
prefix basis:
  Huffman-coded component-codebook level symbols

top-k basis:
  Huffman-coded selected basis indices
  + Huffman-coded component-codebook level symbols
```

The codebook tables and Huffman tables are not counted unless a future variant
adapts them per image.

## Implementation

Added fit-time priors:

```text
control_huffman_priors:
  p95_b4_codebook
  p99_b4_codebook

sparse_topk_control_priors:
  topk_c64_k8_p95_b4_codebook
  ...
```

Each prior stores:

```text
basis_range_mode: component_codebook
codebook_key: lloyd_b4
quantization_mae / quantization_rmse
mean_payload_bytes / min / max
prefix_mean_payload_bytes for prefix controls
static Huffman tables
```

`inspect_stage5_control_basis.py` now emits `codec=huffman` rows for
`basis_range_mode=component_codebook` when matching priors exist.

`eval_stage4_cod_lite_adapter.py` reads the prior metadata and uses
`ComponentCodebookControlCode` for dequantization, while counting exact Huffman
payload bytes per image.

## Promotion Rule

Component-codebook Huffman controls should be screened against:

- fixed-bit component codebook controls
- fixed-bit component p95/p99 controls
- uniform / mu-law Huffman controls
- the current detail-FiLM Stage 4 anchor

Promotion still requires decoded-image metrics, exact actual payload bpp, and
anchor-aware full552 validation. A lower planned mean payload is not sufficient
by itself.
