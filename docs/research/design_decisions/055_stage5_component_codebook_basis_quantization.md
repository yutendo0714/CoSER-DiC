# 055 Stage 5 Component Codebook Basis Quantization

Date: 2026-06-29

## Decision

Add a fixed decoder-side component codebook quantizer for Stage 5 basis
condition-control coefficients.

This is a mainline Stage 5 control-stream improvement, not RGB output
postprocessing.

## Motivation

The previous component-wise basis quantizer used fixed p95/p99 ranges per
basis component. That improves clipping and step size at unchanged payload
bytes, but the quantization levels are still uniformly spaced.

Basis coefficients are usually peaked, skewed, and component-dependent. A
per-component scalar codebook fitted from non-eval train-cache coefficients can
place fixed-bit levels where the coefficients actually occur.

## Bitstream Rule

The codebooks are fixed decoder/model state fitted offline on non-eval data.
They are not transmitted per image.

Per image, CoSER-DiC transmits only the fixed-bit coefficient level indices:

```text
actual_payload_bpp += bits_per_level_index * transmitted_coefficients / pixels
```

For sparse top-k basis controls, the transmitted payload remains:

```text
top-k basis index stream + top-k value-level stream
```

Both streams are counted. The codebook tables are not counted unless a future
variant adapts them per image.

## Implementation

Added:

```text
ComponentCodebookControlCode
--component-codebook-bits
--component-codebook-max-iter
--control-basis-range-mode component_codebook
--range-mode component_codebook
```

The fitter stores:

```text
coefficient_component_codebooks:
  lloyd_b4:
    codec: component_codebook_control_fixed_bits
    method: lloyd_1d_per_component
    bits / levels / component_count
    fixed_bytes_per_image
    quantization_mae / quantization_rmse / max_abs_error
    codebooks
```

The eval path reconstructs the same fixed codebook from the control-basis
checkpoint. In top-k mode, the codec selects the codebook rows corresponding to
the transmitted basis indices before quantizing/dequantizing the selected
values.

## Mainline Use

Use this in the post-affine basis screen together with:

```text
global
component_p95
component_p99
component_codebook
```

Promotion remains metric-based:

- exact `actual_payload_bpp`
- full decoded image metrics
- anchor-aware selection against the detail-FiLM Stage 4 anchor
- no eval-split leakage
- no uncounted per-image side information

This is intended to improve the Stage 5 condition-control quality at the same
payload size before moving to larger counted control streams or decoder LoRA.
