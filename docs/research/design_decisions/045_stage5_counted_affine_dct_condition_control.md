# 045 Stage 5 Counted Affine + DCT Condition Control

Date: 2026-06-29

## Decision

Add a counted Stage 5 condition-control mode:

```text
condition_residual_affine_dct
```

This mode is a mainline CoSER-DiC information-flow change. It modifies the
CoD-Lite condition before decoding and counts all image-specific control bytes
in `actual_payload_bpp`.

It is not final RGB output blending or cosmetic postprocessing.

## Motivation

`condition_residual_affine` can correct group-wise scale and bias mismatch in:

```text
target_cond - base_cond ~= gain * (pred_cond - base_cond) + bias
```

But after the affine correction, there can still be spatial low-frequency
condition error. A very small DCT residual stream can target that remaining
error more directly than another fixed alpha or RGB refiner.

The intended decoder path is:

```text
CoSER semantic/detail payload
  -> Stage 4 adapter prediction
  -> counted affine gain/bias condition correction
  -> counted grouped DCT residual condition correction
  -> frozen or lightly adapted CoD-Lite decoder
```

## Method

For each image:

```text
1. Fit and transmit quantized affine gain_delta and bias.
2. Apply decoded affine correction to pred_cond.
3. Compute remaining target_cond - pred_cond_after_affine.
4. Pool remaining error into channel groups and a coarse spatial grid.
5. DCT-code the first K coefficients per group.
6. Decode and add this residual in condition space.
```

Payload:

```text
control_payload_bytes =
  affine_gain_delta_bytes
  + affine_bias_bytes
  + dct_residual_bytes

actual_payload_bpp =
  stage3_actual_payload_bpp + 8 * control_payload_bytes / pixels
```

The encoder uses the reference image only to decide the image-specific control
payload. Those control bytes are transmitted and counted.

## New CLI

```text
--counted-control-mode condition_residual_affine_dct
--control-affine-groups <G_affine>
--control-affine-grid-size <S_affine>
--control-affine-gain-range <R_gain>
--control-affine-bias-range <R_bias>
--control-groups <G_dct>
--control-grid-size <S_dct>
--control-dct-coeffs-per-group <K>
--control-bits <B>
--control-range <R_dct>
--control-quantizer uniform|mu_law
--control-scale <scale>
```

The affine and DCT branches can use different group/grid granularities.

## Generated Limit64 Plan

Generated pending screen:

```text
results/stage5_counted_control/20260629_detailfilm_affinedct_limit64_sweep_plan.json
results/stage5_counted_control/20260629_detailfilm_affinedct_limit64_sweep_plan.sh
```

Settings:

```text
count: 8
extra control bpp min: 0.00048828125
extra control bpp max: 0.00146484375
anchor: 20260629_stage4_fast8192_detailfilm_ft800_b8_limit64_eval
```

The plan includes GPU preflight and must be run only after CUDA/NVML recovers:

```bash
bash results/stage5_counted_control/20260629_detailfilm_affinedct_limit64_sweep_plan.sh
```

After the sweep:

```bash
.venv/bin/python scripts/select_stage5_control_candidates.py \
  --sweep-plan results/stage5_counted_control/20260629_detailfilm_affinedct_limit64_sweep_plan.json \
  --anchor-summary results/stage4_cod_lite_adapter_eval/20260629_stage4_fast8192_detailfilm_ft800_b8_limit64_eval/summary.json \
  --fail-on-anchor-count-mismatch \
  --max-psnr-drop 0.8 \
  --max-anchor-psnr-drop 0.25 \
  --max-anchor-ms-ssim-drop 0.01 \
  --max-anchor-lpips-increase 0.0 \
  --max-anchor-dists-increase 0.0 \
  --output-json results/stage5_selection/20260629_detailfilm_affinedct_limit64_anchor_selection.json
```

## Promotion Rule

Promote only if:

```text
candidate count matches the limit64 anchor
LPIPS and DISTS improve or match the detail-FiLM anchor
PSNR/MS-SSIM remain within anchor guard
actual_payload_bpp includes all affine and DCT control bytes
condition_l1 improvement translates into final decoded image metrics
```

If promoted, run full552 before claiming any external win. BD-rate claims still
require multiple CoSER operating points and official baseline curves.

## Current Status

GPU is still unavailable:

```text
nvidia-smi: Failed to initialize NVML: Unknown Error
torch.cuda.is_available: false
```

No GPU evaluation has been launched.

CPU verification:

```text
tests/test_eval_stage4_cod_lite_adapter_control.py
tests/test_sweep_stage5_counted_control.py
tests/test_select_stage5_control_candidates.py

28 passed
```

