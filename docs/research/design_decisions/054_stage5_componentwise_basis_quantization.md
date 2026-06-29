# Decision 054: Stage 5 Component-Wise Basis Quantization

Date: 2026-06-29 JST

## Status

Implemented in entropy/evaluation/planning code. CPU tests passed.

GPU execution has not been launched because CUDA/NVML is currently unavailable.

## Motivation

The Stage 5 affine+basis control stream sends a small number of basis
coefficients for the CoD-Lite condition residual. The previous fixed-bit path
used one global scalar range for all coefficients:

```text
codec=fixed_bits
range=p95 or p99 global coefficient quantile
```

But PCA/SVD basis coefficients have different per-component scales. A single
global range wastes quantization levels on small components or clips large
components.

This is a direct Stage 5 bottleneck:

```text
same transmitted coefficient count
same actual payload bytes
different condition recovery quality
```

## Decision

Add fixed decoder-side per-component ranges for fixed-bit basis controls:

```text
basis_range_mode=global
basis_range_mode=component_p95
basis_range_mode=component_p99
```

The ranges come from non-eval train-cache coefficient statistics already stored
in the basis checkpoint:

```text
coefficient_component_abs_p95
coefficient_component_abs_p99
```

At test time, no extra side information is transmitted. The range table is
fixed decoder/model state, like the basis itself.

## Payload Accounting

For prefix fixed-bit basis controls, the payload byte count is unchanged:

```text
payload_bits = components * bits
payload_bytes = ceil(payload_bits / 8)
```

Only the quantizer dequantization scale changes per coefficient. Therefore
`actual_payload_bpp` remains comparable to the global-range fixed-bit basis
setting.

For sparse top-k fixed-bit basis controls, the decoder first decodes the
selected basis indices, then uses those indices to select the corresponding
fixed component ranges before decoding the value stream:

```text
index stream -> selected component ids
value stream -> dequantize values with range[selected component id]
```

The payload byte count is still:

```text
index bytes + value bytes
```

with no extra per-image side information.

## Scope

Enabled:

```text
codec=fixed_bits
selection=prefix
selection=topk
mode=basis
mode=affine_basis
mode=hybrid_affine_dct_grid_basis
```

Not enabled yet:

```text
codec=huffman
```

Huffman requires component-conditioned entropy priors to avoid optimistic
payload accounting. Keep Huffman global-range until those priors are fitted
explicitly.

## Implementation

```text
src/coserdic/entropy/control_grid.py
  ComponentUniformControlCode
  ComponentMuLawControlCode

scripts/eval_stage4_cod_lite_adapter.py
  --control-basis-range-mode global|component_p95|component_p99
  --control-basis-range-floor

scripts/inspect_stage5_control_basis.py
  --range-mode global
  --range-mode component_p95
  --range-mode component_p99

scripts/sweep_stage5_counted_control.py
  basis_range_mode=<mode>
```

The post-affine basis prepare plan now asks inspect to emit all three range
modes:

```text
results/stage5_counted_control/20260629_detailfilm_postaffine_basis_prepare_plan.sh
```

## Promotion Rule

Component-wise quantization is a candidate improvement, not a result.

Promote only if it improves decoded-image metrics or anchor-aware selection at
the same actual payload bytes:

```text
LPIPS / DISTS improve
PSNR / MS-SSIM guard holds
full552 confirms the limit64 signal
official baseline comparison remains protocol matched
```

If component-wise range improves condition L1 but not decoded-image metrics, it
should remain a diagnostic.
