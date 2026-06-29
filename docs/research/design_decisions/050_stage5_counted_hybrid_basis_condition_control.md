# Decision 050: Stage 5 Counted Hybrid Basis Condition Control

Date: 2026-06-29 JST

## Status

Implemented and wired into the post-affine basis prepare plan.

GPU execution has not been launched because CUDA/NVML is currently unavailable.

## Motivation

Post-affine basis control can represent condition residual structure that a
small affine correction cannot. However, sending basis coefficients for every
image may waste bits on easy images where `none` or `affine` is enough.

For Stage 5 BD-rate, this matters:

```text
same perceptual gain with fewer bits
or larger perceptual gain only on images that need the extra control
```

## Decision

Add a basis-aware hybrid mode:

```text
condition_residual_hybrid_affine_dct_grid_basis
```

For each image, the encoder evaluates a condition-space RD proxy over:

```text
none
affine
affine + DCT
affine + grid
affine + basis
```

and transmits:

```text
mode selector bits/bytes
+ selected control payload bytes
```

The selector is image-specific and counted. The basis itself, static Huffman
priors, and fixed quantizer/compander parameters are decoder-side model state.

There are five hybrid-basis modes, so compact selector accounting uses:

```text
selector_bits = 3
```

when exact fixed-bit payload sizes are available. Huffman/top-k variable-length
payloads fall back to byte-granular selector accounting.

## Huffman Support

This mode supports the same basis entropy options as `affine_basis`:

```text
fixed_bits
huffman
uniform
mu_law
prefix basis coefficients
sparse top-k basis coefficients
```

Decision 054 adds fixed-bit basis range modes:

```text
global
component_p95
component_p99
```

The component ranges are fixed decoder-side basis statistics and do not add
image-specific bits. For sparse top-k fixed-bit basis, decoded indices select
the corresponding fixed component ranges before value dequantization.

During planning, huffman basis payload uses train-cache mean payload bytes.
The actual evaluation still counts the realized per-image payload bytes from
encode/decode, not the planning estimate.

For conservative planning, the reported `control_bpp` is:

```text
max(
  selector-aware none bytes,
  selector-aware affine bytes,
  selector-aware affine + DCT bytes,
  selector-aware affine + grid bytes,
  selector-aware affine + basis mean bytes
)
```

The final paper metric is not this planning number. It is the actual per-image
selected payload:

```text
actual_payload_bpp =
  Stage 3 semantic/detail payload bpp
  + selected control payload bpp
```

## Implementation

```text
scripts/eval_stage4_cod_lite_adapter.py
  --counted-control-mode condition_residual_hybrid_affine_dct_grid_basis

scripts/sweep_stage5_counted_control.py
  mode=hybrid_affine_dct_grid_basis
  --override-mode hybrid_affine_dct_grid_basis
  --override-hybrid-selector-bits 3
```

The override mode converts inspected `mode=basis` settings into hybrid-basis
settings after the post-affine basis has been fitted.

## Generated Plan

The post-affine basis prepare shell now generates:

```text
results/stage5_counted_control/20260629_detailfilm_hybridbasis_limit64_sweep_plan.json
results/stage5_counted_control/20260629_detailfilm_hybridbasis_limit64_sweep_plan.sh
```

The master Stage 5 screen runs that generated shell after the basis prepare
step:

```text
results/stage5_counted_control/20260629_stage5_mainline_screen_select_plan.sh
```

## Promotion Rule

Do not promote this mode because it improves condition L1 alone.

Promotion still requires:

```text
anchor-aware limit64 selection
actual_payload_bpp accounting
PSNR / MS-SSIM guard
LPIPS / DISTS improvement
full552 confirmation
protocol-matched curve comparison before BD-rate claims
```

If `control_hybrid_mode_counts` collapses to one mode, the corresponding
single-family control should be preferred unless the selector overhead is
negligible and full552 metrics justify it.
