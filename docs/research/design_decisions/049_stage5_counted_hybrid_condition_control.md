# Decision 049: Stage 5 Counted Hybrid Condition Control

Date: 2026-06-29 JST

## Status

Implemented and planned for the next GPU screen.

GPU execution has not been launched because CUDA/NVML is currently unavailable.

## Motivation

The previous counted-control families each use one fixed condition-control
structure for every image:

```text
affine
affine + DCT
affine + grid
affine + basis
```

This is too rigid for Stage 5. Some images may only need a cheap group-wise
scale/bias correction, while others need spatial residual control. Forcing one
control family across all images can waste bits on easy images or
under-control hard images.

## Decision

Add a counted hybrid mode:

```text
condition_residual_hybrid_affine_dct_grid
```

For each image, the encoder evaluates a condition-space RD proxy over:

```text
none
affine
affine + DCT
affine + grid
```

and transmits:

```text
mode selector bytes
+ selected control payload bytes
```

The decoder receives the selector and selected payload, then applies the same
condition-space correction before the frozen or adapted CoD-Lite decoder
backbone. This is not RGB output postprocessing.

## Selection Score

Current proxy:

```text
score = condition_l1_after_control
        + control_hybrid_rate_lambda * selected_control_bpp
```

The default `control_hybrid_rate_lambda=0` selects the best condition recovery.
Positive lambda values trade some condition accuracy for fewer transmitted
control bytes.

## Payload Policy

The selector is image-specific and must be counted. The current implementation
uses a conservative default:

```text
control_hybrid_selector_bytes = 1 byte / image
```

This is larger than the theoretical packed selector for four modes, but avoids
optimistic paper bpp while the final production bitstream is not implemented.

Decision 052 adds optional exact selector-bit accounting:

```text
selector_bits=2 for the four-mode hybrid selector
```

When exact fixed-bit payload sizes are known, eval/planning can count:

```text
ceil((payload_bits + selector_bits) / 8)
```

Variable-length Huffman/top-k candidates still fall back to byte-granular
selector accounting to avoid undercounting.

For paper metric:

```text
actual_payload_bpp =
  Stage 3 semantic/detail payload bpp
  + 8 * (selector bytes + selected control payload bytes) / pixels
```

## Implementation

```text
scripts/eval_stage4_cod_lite_adapter.py
  --counted-control-mode condition_residual_hybrid_affine_dct_grid
  --control-hybrid-selector-bytes
  --control-hybrid-selector-bits
  --control-hybrid-rate-lambda

scripts/sweep_stage5_counted_control.py
  mode=hybrid_affine_dct_grid
  selector_bytes=<int>
  selector_bits=<int>
  rd_lambda=<float>
```

Per-image metrics include:

```text
control_hybrid_mode
control_hybrid_mode_index
control_hybrid_condition_l1
control_payload_bytes
control_payload_bpp
actual_payload_bpp
```

The summary records `control_hybrid_mode_counts`.

## Planned Screen

```text
results/stage5_counted_control/20260629_detailfilm_hybrid_limit64_sweep_plan.json
results/stage5_counted_control/20260629_detailfilm_hybrid_limit64_sweep_plan.sh
```

The generated screen has seven candidates. Planned conservative control bpp is
approximately:

```text
0.000763 to 0.002228 bpp at 512x512
```

The preferred master screen includes this plan:

```text
results/stage5_counted_control/20260629_stage5_mainline_screen_select_plan.sh
```

## Promotion Rule

Hybrid candidates are not promoted by condition L1 alone.

They must pass the same anchor-aware selection as all other Stage 5 candidates:

```text
compare against the detail-FiLM limit64 anchor
preserve actual_payload_bpp accounting
preserve PSNR / MS-SSIM fidelity guards
improve LPIPS / DISTS under the configured guards
promote only guarded candidates to full552
```

## Expected Benefit

If image-specific control type matters, hybrid mode should:

```text
match or beat the best single family in condition recovery
spend fewer bits on images where none or affine is enough
reserve grid/DCT control for harder images
produce a better rate-perception curve than any single control family
```

If the selector always chooses one family, that is still useful evidence: the
single-family path should be preferred and hybrid should not be promoted.
