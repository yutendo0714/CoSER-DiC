# Decision 051: Stage 5 Hybrid Image-RDO Selector

Date: 2026-06-29 JST

## Status

Implemented and planned for the next GPU screen.

GPU execution has not been launched because CUDA/NVML is currently unavailable.

## Motivation

The first hybrid selectors choose the transmitted control family by condition
L1:

```text
score = condition_l1_after_control + lambda * control_bpp
```

That is a useful proxy, but Stage 5 must win on decoded image metrics:

```text
LPIPS
DISTS
PSNR / MS-SSIM guards
eventually BD-rate under protocol-matched curves
```

Condition L1 can choose a condition tensor that is closer to the native
CoD-Lite condition but not best for the final decoded image. Therefore Stage 5
needs an encoder-side RDO selector that actually runs candidate condition
payloads through the decoder and chooses by image-quality proxy.

## Decision

Add:

```text
--control-hybrid-selection-objective
--control-hybrid-fidelity-lambda
--control-hybrid-fidelity-metric
```

with:

```text
condition_l1
image_l1
image_mse
lpips_alex
dists
```

For image objectives, the encoder evaluates each candidate control family by:

```text
1. encode/decode candidate control payload
2. form candidate condition tensor
3. run the fixed/adapted CoD-Lite decoder
4. score the decoded image against the original reference
5. transmit only the selected mode selector and selected payload bytes
```

This is legal codec RDO. The encoder may inspect the original image when
choosing coding mode. The decoder does not receive the original image, metric
scores, or unselected candidates.

Decision 053 adds an optional fidelity guard for image objectives:

```text
score = image_objective
        + fidelity_lambda * image_l1_or_mse
        + rate_lambda * selected_control_bpp
```

This is intended to reduce LPIPS/DISTS-only choices that damage fidelity or
structure.

## Payload Accounting

The selector and selected payload bytes are counted:

```text
actual_payload_bpp =
  Stage 3 semantic/detail payload bpp
  + 8 * (selector bytes + selected control payload bytes) / pixels
```

Unselected candidate payloads are not transmitted and are not counted.

Decision 052 refines this for fixed-bit candidate streams:

```text
selector_bits > 0:
  use exact packed selector bits when payload_bits are known

unknown variable-length payload bits:
  fall back to selector_bytes
```

The image-RDO score and final per-image `control_payload_bytes` use the same
selector-aware payload calculation.

## Implementation

```text
scripts/eval_stage4_cod_lite_adapter.py
  --control-hybrid-selection-objective
  --control-hybrid-selector-bits
  --control-hybrid-fidelity-lambda
  --control-hybrid-fidelity-metric
  control_hybrid_rdo_score in per-image metrics

scripts/sweep_stage5_counted_control.py
  objective=<condition_l1|image_l1|image_mse|lpips_alex|dists>
  selector_bits=<int>
  fidelity_lambda=<float>
  fidelity_metric=<image_l1|image_mse>
```

The image-RDO path is currently restricted to fixed output blend-alpha without
learned output gates. This avoids ambiguity where the selector would also have
to evaluate a learned image-space gate.

## Generated Plans

Non-basis image-RDO hybrid screen:

```text
results/stage5_counted_control/20260629_detailfilm_hybrid_imagerdo_limit64_sweep_plan.json
results/stage5_counted_control/20260629_detailfilm_hybrid_imagerdo_limit64_sweep_plan.sh
```

Guarded non-basis image-RDO screen:

```text
results/stage5_counted_control/20260629_detailfilm_hybrid_imagerdo_guarded_limit64_sweep_plan.json
results/stage5_counted_control/20260629_detailfilm_hybrid_imagerdo_guarded_limit64_sweep_plan.sh
```

Post-affine basis prepare now generates DISTS-RDO hybrid-basis screen:

```text
results/stage5_counted_control/20260629_detailfilm_hybridbasis_distsrdo_limit64_sweep_plan.json
results/stage5_counted_control/20260629_detailfilm_hybridbasis_distsrdo_limit64_sweep_plan.sh
```

Both are included in:

```text
results/stage5_counted_control/20260629_stage5_mainline_screen_select_plan.sh
```

## Promotion Rule

Image-RDO selectors are stronger and more expensive, but they are not automatic
Stage 5 wins.

Promotion still requires:

```text
anchor-aware limit64 selection
actual_payload_bpp accounting from selected bytes
PSNR / MS-SSIM guard
LPIPS / DISTS improvement
full552 confirmation
protocol-matched official baseline curve before BD-rate claims
```

If `lpips_alex` or `dists` RDO overfits the limit64 subset or harms fidelity,
it should be rejected even if its selected objective improves.
