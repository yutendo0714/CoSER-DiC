# Decision 053: Stage 5 Guarded Image-RDO Selector

Date: 2026-06-29 JST

## Status

Implemented and planned for the next GPU screen.

GPU execution has not been launched because CUDA/NVML is currently unavailable.

## Motivation

Image-RDO hybrid selection lets the encoder choose the transmitted condition
control family by decoded image quality rather than condition L1 alone.

However, pure perceptual objectives can be too aggressive:

```text
LPIPS / DISTS improves
but structure, PSNR, or MS-SSIM can regress
```

Stage 5 should improve rate-perception while preserving the CoSER
semantic/detail structure advantage. The selector therefore needs a fidelity
guard inside the encoder-side RDO score, not an RGB postprocess after decoding.

## Decision

Extend image-RDO hybrid selection with:

```text
--control-hybrid-fidelity-lambda
--control-hybrid-fidelity-metric image_l1|image_mse
```

For image objectives, the per-candidate score becomes:

```text
score =
  image_objective(reference, decoded_candidate)
  + fidelity_lambda * fidelity_metric(reference, decoded_candidate)
  + rate_lambda * selected_control_bpp
```

The decoder still receives only:

```text
mode selector
+ selected entropy-coded control payload
```

The reference image, objective score, and unselected candidates are not
transmitted.

## Implementation

```text
scripts/eval_stage4_cod_lite_adapter.py
  --control-hybrid-fidelity-lambda
  --control-hybrid-fidelity-metric

scripts/sweep_stage5_counted_control.py
  fidelity_lambda=<float>
  fidelity_metric=<image_l1|image_mse>
```

The per-image metric `control_hybrid_rdo_score` stores the final guarded score.

## Generated Plan

```text
results/stage5_counted_control/20260629_detailfilm_hybrid_imagerdo_guarded_limit64_sweep_plan.json
results/stage5_counted_control/20260629_detailfilm_hybrid_imagerdo_guarded_limit64_sweep_plan.sh
```

The plan covers:

```text
DISTS + image_l1 guard
DISTS + image_mse guard
LPIPS-Alex + image_l1 guard
LPIPS-Alex + image_mse guard
selector_bits=2 for the four-mode hybrid selector
```

It is included in:

```text
results/stage5_counted_control/20260629_stage5_mainline_screen_select_plan.sh
```

## Promotion Rule

Guarded image-RDO is not automatically better than unguarded image-RDO.

Promotion requires:

```text
anchor-aware limit64 improvement
actual_payload_bpp from selected transmitted bytes
PSNR / MS-SSIM guard
LPIPS / DISTS improvement
visual failure audit
full552 confirmation
protocol-matched official baseline comparison
```

If the guarded selector simply chooses the same mode distribution as a cheaper
condition-L1 selector, the cheaper selector should be preferred.
