# 057 Stage 5 Vector Codebook Basis Control

Date: 2026-06-29

## Decision

Add vector-codebook basis control as a low-bpp Stage 5 condition-control
family.

Instead of transmitting K scalar basis coefficients, this path transmits one
codebook index for the whole K-dimensional coefficient prefix:

```text
condition residual grid
  -> fixed basis projection
  -> first K coefficients
  -> fixed decoder-side vector codebook
  -> one transmitted vector index
```

## Motivation

The current scalar prefix/top-k basis controls are expressive, but they spend
bits independently per coefficient. For BD-rate curves, CoSER also needs very
low-control-bpp operating points that may improve perceptual metrics slightly
without adding many bytes.

Vector quantization gives a different tradeoff:

- less precise than scalar coefficient coding
- much smaller payload at K > 1
- can capture common correlated coefficient patterns
- useful as a low-rate curve point and as a hybrid candidate

This keeps the CoSER mainline intact because the transmitted payload is still
an explicit entropy-coded condition-control stream, not posthoc RGB editing.

## Bitstream Rule

The vector codebook is fixed decoder-side state fitted from non-eval
train-cache coefficients.

Per image, count only:

```text
fixed_bits:
  one fixed-bit vector index

huffman:
  one static-Huffman-coded vector index
```

The vector table and Huffman table are not counted unless a future variant
adapts them per image.

## Implementation

Added:

```text
VectorCodebookControlCode
--vector-codebook-components
--vector-codebook-bits
--vector-codebook-max-iter
--control-basis-selection vector
```

Fit output:

```text
coefficient_vector_codebooks:
  vq_k8_b8:
    codec: vector_codebook_control_fixed_bits
    selection: vector
    components: 8
    bits: 8
    fixed_bytes_per_image
    huffman_mean_payload_bytes
    quantization_mae / quantization_rmse
    vectors
    huffman
```

Inspect output emits rows like:

```text
mode=basis,...,coeffs=8,bits=8,codec=fixed_bits,selection=vector
mode=basis,...,coeffs=8,bits=8,codec=huffman,selection=vector,huffman_key=vq_k8_b8
```

The post-affine basis prepare plan now fits:

```text
--vector-codebook-components 4 8 16
--vector-codebook-bits 4 6 8
```

## Promotion Rule

Vector-codebook controls are not expected to dominate scalar basis controls at
the same control bpp. Their purpose is to add low-rate curve points and hybrid
candidates.

Promote only if decoded-image metrics improve against the detail-FiLM anchor
after exact payload accounting. Planned retained energy or train-cache
quantization error alone is not enough.
