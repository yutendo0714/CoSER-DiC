# 046 Stage 5 Counted Affine + Grid Condition Control

Date: 2026-06-29

## Decision

Add a counted Stage 5 condition-control mode:

```text
condition_residual_affine_grid
```

This is a CoSER-owned condition-space control stream for the CoD-Lite decoder.
It is not RGB postprocessing.

## Motivation

`condition_residual_affine_dct` spends very few bits on low-frequency residual
condition error after affine correction. That is efficient, but the DCT prefix
can miss spatial residuals whose energy is not concentrated in the first few
zig-zag coefficients.

`condition_residual_affine_grid` is the higher-flexibility sibling:

```text
1. Fit and transmit quantized affine gain/bias correction.
2. Apply decoded affine correction to pred_cond.
3. Pool remaining condition error into grouped spatial grids.
4. Transmit the quantized grid residual.
5. Decode and add it before the CoD-Lite decoder.
```

This spends slightly more payload than affine+DCT, but it may be a better
Stage-5 candidate if CoD-Lite perceptual gap is caused by spatially structured
condition mismatch.

## Payload

```text
control_payload_bytes =
  affine_gain_delta_bytes
  + affine_bias_bytes
  + grouped_grid_residual_bytes

actual_payload_bpp =
  stage3_actual_payload_bpp + 8 * control_payload_bytes / pixels
```

All image-specific control bytes are transmitted and counted. Fixed model
weights, fixed adapter weights, and fixed CoD-Lite decoder weights are not.

## New CLI

```text
--counted-control-mode condition_residual_affine_grid
--control-affine-groups <G_affine>
--control-affine-grid-size <S_affine>
--control-affine-gain-range <R_gain>
--control-affine-bias-range <R_bias>
--control-groups <G_grid>
--control-grid-size <S_grid>
--control-bits <B>
--control-range <R_grid>
--control-quantizer uniform|mu_law
--control-scale <scale>
```

## Generated Limit64 Plan

Generated pending screen:

```text
results/stage5_counted_control/20260629_detailfilm_affinegrid_limit64_sweep_plan.json
results/stage5_counted_control/20260629_detailfilm_affinegrid_limit64_sweep_plan.sh
```

Settings:

```text
count: 8
extra control bpp min: 0.0003662109375
extra control bpp max: 0.002197265625
anchor: 20260629_stage4_fast8192_detailfilm_ft800_b8_limit64_eval
```

Run only after CUDA/NVML recovers:

```bash
bash results/stage5_counted_control/20260629_detailfilm_affinegrid_limit64_sweep_plan.sh
```

Then select against the matching limit64 detail-FiLM anchor:

```bash
.venv/bin/python scripts/select_stage5_control_candidates.py \
  --sweep-plan results/stage5_counted_control/20260629_detailfilm_affinegrid_limit64_sweep_plan.json \
  --anchor-summary results/stage4_cod_lite_adapter_eval/20260629_stage4_fast8192_detailfilm_ft800_b8_limit64_eval/summary.json \
  --fail-on-anchor-count-mismatch \
  --max-psnr-drop 0.8 \
  --max-anchor-psnr-drop 0.25 \
  --max-anchor-ms-ssim-drop 0.01 \
  --max-anchor-lpips-increase 0.0 \
  --max-anchor-dists-increase 0.0 \
  --output-json results/stage5_selection/20260629_detailfilm_affinegrid_limit64_anchor_selection.json
```

## Promotion Rule

Promote only if:

```text
candidate count matches the limit64 anchor
LPIPS and DISTS improve or match the detail-FiLM anchor
PSNR/MS-SSIM remain within anchor guard
actual_payload_bpp includes all affine and grid residual bytes
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

No GPU evaluation has been launched.

CPU verification:

```text
tests/test_eval_stage4_cod_lite_adapter_control.py
tests/test_sweep_stage5_counted_control.py

18 passed
```

