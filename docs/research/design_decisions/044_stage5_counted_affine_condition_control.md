# 044 Stage 5 Counted Affine Condition Control

Date: 2026-06-29

## Decision

Add a counted Stage 5 condition-control mode:

```text
condition_residual_affine
```

This is a tiny image-specific payload that modifies the CoD-Lite condition
before decoding. It does not blend final RGB outputs.

## Motivation

Current CoSER Stage 4 improves over Stage 3, but official CoD-Lite perceptual
metrics are still far ahead. Additive grid/DCT/basis control can send a
condition residual, but a very low-rate control stream may be more effective if
it corrects the adapter residual itself:

```text
base_cond + adapter_residual
  -> base_cond + affine_gain * adapter_residual + affine_bias
  -> CoD-Lite decoder
```

The decoder can reproduce `base_cond` and `adapter_residual` from fixed model
state plus the CoSER semantic/detail payload. Only quantized affine gain deltas
and biases are transmitted.

## Method

For each image, channel group, and optional coarse spatial cell, the encoder
fits:

```text
target_cond - base_cond ~= gain * (pred_cond - base_cond) + bias
```

It transmits:

```text
gain_delta = gain - 1
bias
```

The eval path simulates true transmission by encoding and decoding both tensors
before applying the correction.

New CLI:

```text
--counted-control-mode condition_residual_affine
--control-groups <G>
--control-grid-size <S>
--control-bits <B>
--control-affine-gain-range <R_gain>
--control-affine-bias-range <R_bias>
--control-quantizer uniform|mu_law
--control-scale <scale>
```

Payload:

```text
control_payload_bytes =
  bytes(gain_delta GxSxS tensor) + bytes(bias GxSxS tensor)

actual_payload_bpp =
  stage3_actual_payload_bpp + 8 * control_payload_bytes / pixels
```

## Current Plan

Generated limit64 plan:

```text
results/stage5_counted_control/20260629_detailfilm_affine_limit64_sweep_plan.json
results/stage5_counted_control/20260629_detailfilm_affine_limit64_sweep_plan.sh
```

Candidate payload range:

```text
min control bpp: 0.000244140625
max control bpp: 0.001953125
```

Settings:

```text
groups: 8 / 16
grid: 1 / 2
bits: 4
gain range: 1.0 uniform, 1.5 mu-law
bias range: 0.25
```

The initial eight-setting plan was reduced into a cleaner six-setting
curve-oriented limit64 plan, with two candidates per extra-control-bpp band:

```text
results/stage5_counted_control/20260629_detailfilm_affine_limit64_curve_settings.json
results/stage5_counted_control/20260629_detailfilm_affine_curve_limit64_sweep_plan.json
results/stage5_counted_control/20260629_detailfilm_affine_curve_limit64_sweep_plan.sh
```

This uses the same counted affine mechanism, but avoids spending GPU time on
redundant same-bpp settings before the first anchor-guarded screen.

Run after GPU restart:

```bash
bash results/stage5_counted_control/20260629_detailfilm_affine_curve_limit64_sweep_plan.sh
```

Then select against the matching limit64 detail-FiLM anchor:

```bash
.venv/bin/python scripts/select_stage5_control_candidates.py \
  --sweep-plan results/stage5_counted_control/20260629_detailfilm_affine_curve_limit64_sweep_plan.json \
  --anchor-summary results/stage4_cod_lite_adapter_eval/20260629_stage4_fast8192_detailfilm_ft800_b8_limit64_eval/summary.json \
  --fail-on-anchor-count-mismatch \
  --max-psnr-drop 0.8 \
  --max-anchor-psnr-drop 0.25 \
  --max-anchor-ms-ssim-drop 0.01 \
  --max-anchor-lpips-increase 0.0 \
  --max-anchor-dists-increase 0.0 \
  --output-json results/stage5_selection/20260629_detailfilm_affine_limit64_anchor_selection.json
```

## Promotion Rule

Promote only if the candidate:

```text
matches limit64 count
beats or matches the detail-FiLM limit64 anchor on LPIPS and DISTS
does not exceed anchor fidelity guards
counts the affine payload in actual_payload_bpp
improves final decoded image metrics, not only condition L1
```

If the affine probe works, the next step is to combine:

```text
LoRA/partial backbone adaptation
+ counted affine control
+ optional additive basis/DCT residual control
```

and build multi-rate full552 curves before any BD-rate claim.

## Verification

CPU tests:

```bash
PYTHONPATH=/workspace/CoSER-DiC .venv/bin/pytest \
  tests/test_eval_stage4_cod_lite_adapter_control.py \
  tests/test_sweep_stage5_counted_control.py \
  tests/test_select_stage5_control_candidates.py -q
```

Result:

```text
24 passed
```

Additional CPU continuation:

```text
scripts/sweep_stage5_counted_control.py now accepts both string and dict
setting rows from generated sweep plans, and `--include-mode` is supported by
the sweep and curve-planning tools. This lets affine/grid/basis settings be
reused without copying commands or leaking stale summary paths into metadata.
```

Verification:

```text
tests/test_sweep_stage5_counted_control.py
tests/test_plan_stage5_control_curve.py

17 passed
```
