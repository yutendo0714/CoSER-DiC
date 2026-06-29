# 047 Stage 5 Counted Affine + Basis Condition Control

Date: 2026-06-29

## Decision

Add a counted Stage 5 condition-control mode:

```text
condition_residual_affine_basis
```

This is a CoSER-owned condition-space control stream for the CoD-Lite decoder.
It is not RGB postprocessing.

## Motivation

The current Stage 5 counted controls form a progression:

```text
affine:
  correct condition residual scale and bias

affine + DCT:
  correct low-frequency residual condition error

affine + grid:
  correct grouped spatial residual condition error

affine + basis:
  correct the remaining condition error in a train-cache-fitted PCA/SVD basis
```

`affine_basis` should be the most bit-efficient member of this family if the
post-affine condition residual has a learnable low-dimensional structure.

## Method

For each image:

```text
1. Fit and transmit quantized affine gain_delta and bias.
2. Apply decoded affine correction to pred_cond.
3. Compute remaining target_cond - pred_cond_after_affine.
4. Pool remaining error into grouped spatial grids.
5. Project onto a fixed decoder-side basis fitted from non-eval train cache.
6. Transmit quantized basis coefficients.
7. Decode and add the reconstructed residual in condition space.
```

Payload:

```text
control_payload_bytes =
  affine_gain_delta_bytes
  + affine_bias_bytes
  + basis_coefficient_bytes

actual_payload_bpp =
  stage3_actual_payload_bpp + 8 * control_payload_bytes / pixels
```

The basis, mean, and Huffman priors are fixed decoder-side state. Per-image
coefficients are transmitted and counted.

## Post-Affine Basis Fit

`scripts/fit_stage5_condition_control_basis.py` now supports fitting the basis
after an affine correction:

```text
--pre-basis-affine true
--pre-basis-affine-groups <G_affine>
--pre-basis-affine-grid-size <S_affine>
--pre-basis-affine-bits <B>
--pre-basis-affine-gain-range <R_gain>
--pre-basis-affine-bias-range <R_bias>
--pre-basis-affine-quantizer uniform|mu_law
```

This matters because a basis fitted to raw adapter error may spend components
on scale/bias mismatch that the affine stream can already handle.

## Generated Prepare Plan

GPU-restart entrypoint:

```bash
bash results/stage5_counted_control/20260629_detailfilm_postaffine_basis_prepare_plan.sh
```

This plan does three things:

```text
1. Fit a post-affine condition residual basis from the 8192-image non-eval
   train cache.
2. Inspect the basis and produce recommended basis settings.
3. Generate a dry-run affine_basis limit64 sweep plan from those settings.
```

Expected outputs after the prepare plan runs:

```text
outputs/stage5_control_basis/20260629_detailfilm_postaffine_basis_g16s8_k64_train8192/control_basis.pt
outputs/stage5_control_basis/20260629_detailfilm_postaffine_basis_g16s8_k64_train8192/recommended_basis_settings.json
results/stage5_counted_control/20260629_detailfilm_affinebasis_limit64_sweep_plan.json
results/stage5_counted_control/20260629_detailfilm_affinebasis_limit64_sweep_plan.sh
```

Then run:

```bash
bash results/stage5_counted_control/20260629_detailfilm_affinebasis_limit64_sweep_plan.sh
```

Then select:

```bash
.venv/bin/python scripts/select_stage5_control_candidates.py \
  --sweep-plan results/stage5_counted_control/20260629_detailfilm_affinebasis_limit64_sweep_plan.json \
  --anchor-summary results/stage4_cod_lite_adapter_eval/20260629_stage4_fast8192_detailfilm_ft800_b8_limit64_eval/summary.json \
  --fail-on-anchor-count-mismatch \
  --max-psnr-drop 0.8 \
  --max-anchor-psnr-drop 0.25 \
  --max-anchor-ms-ssim-drop 0.01 \
  --max-anchor-lpips-increase 0.0 \
  --max-anchor-dists-increase 0.0 \
  --output-json results/stage5_selection/20260629_detailfilm_affinebasis_limit64_anchor_selection.json
```

## Promotion Rule

Promote only if:

```text
candidate count matches the limit64 anchor
LPIPS and DISTS improve or match the detail-FiLM anchor
PSNR/MS-SSIM remain within anchor guard
actual_payload_bpp includes all affine and basis coefficient bytes
final decoded image metrics improve, not just condition_l1
```

If promoted, run full552 and compare against official CoD-Lite/CoD curves
before any external-baseline claim.

## Current Status

GPU is still unavailable:

```text
nvidia-smi: Failed to initialize NVML: Unknown Error
torch.cuda.is_available: false
```

No GPU basis fitting or evaluation has been launched.

CPU verification:

```text
tests/test_eval_stage4_cod_lite_adapter_control.py
tests/test_sweep_stage5_counted_control.py
tests/test_fit_stage5_condition_control_basis.py

27 passed
```

