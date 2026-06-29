# 041 Stage 5 Condition Residual RMS Guard Probe

Date: 2026-06-29

## Decision

Add a deterministic condition-space RMS guard for Stage 4 / Stage 5 CoD-Lite
adapter evaluation.

This is not RGB output postprocessing. It operates before CoD-Lite decoding:

```text
decoded CoSER semantic/detail payload
  -> CoSER condition adapter
  -> base CoD-Lite condition + adapter residual
  -> deterministic RMS guard in condition space
  -> CoD-Lite decoder backbone
```

The guard uses only decoder-available tensors:

```text
base_condition = CoD-Lite native condition from decoded Stage 3 x_aux
condition_residual = adapter_condition - base_condition
```

It does not use the reference image at decode time and does not transmit any
new image-specific side information. Therefore it does not change
`actual_payload_bpp / paper_bpp`.

## Motivation

Current raw adapters can strongly improve LPIPS/DISTS but damage PSNR/MS-SSIM
and visual structure. Fixed RGB alpha blends are useful diagnostics but should
not become the method.

The mainline hypothesis is that part of the damage comes from destructive
condition residual magnitude or distribution mismatch. A condition-space guard
is a cleaner probe than final-output blending because it tests whether
stabilizing the adapter residual before the diffusion backbone improves the
rate-perception-faithfulness tradeoff.

## Implementation

Added:

```text
src/coserdic/models/gencodec_backbone.py
  condition_residual_rms_guard

scripts/eval_stage4_cod_lite_adapter.py
  --condition-residual-guard none|rms_clip
  --condition-residual-guard-granularity global|spatial|channel
  --condition-residual-max-rms-ratio
  --condition-residual-min-gate

scripts/sweep_stage5_condition_residual_guard.py
  screen/full552 command-plan generation

scripts/select_stage5_control_candidates.py
  preserves condition_residual_guard metadata from sweep plans

scripts/train_stage4_cod_lite_adapter.py
  --condition-residual-rms-guard-weight
  --condition-residual-rms-guard-ratio
  --condition-residual-rms-guard-granularity
```

Guard behavior:

```text
base_rms = rms(base_condition)
residual_rms = rms(condition_residual)
max_residual_rms = max_rms_ratio * base_rms
gate = min(1, max_residual_rms / residual_rms)
guarded_residual = gate * condition_residual
```

Supported granularities:

```text
global:
  one gate per image

channel:
  one gate per image and condition channel

spatial:
  one gate per image and spatial condition location
```

Eval rows and summaries now record:

```text
condition_residual_guard_mean
condition_residual_guard_min
condition_residual_guard_max
condition_residual_guard
condition_residual_guard_granularity
condition_residual_max_rms_ratio
condition_residual_min_gate
```

The training script also supports an optional RMS-excess loss:

```text
loss += weight * mean(relu(residual_rms / base_rms - max_ratio)^2)
```

This teaches the adapter to avoid destructive condition residuals rather than
only clipping them at evaluation time. The default weight is `0.0`, so existing
checkpoints remain reproducible.

## Generated Plans

Current strongest no-extra-bit full552 Stage 4 anchor:

```text
checkpoint:
  checkpoints/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailfilm_ft800_b8.pt

full552 reference result:
  actual_payload_bpp: 0.013999109682829483
  PSNR: 21.2427
  MS-SSIM: 0.7150
  LPIPS: 0.4304
  DISTS: 0.2982
```

Generated guard sweep plans:

```text
results/stage5_condition_guard/20260629_detailfilm_guard_screen64_plan.json
results/stage5_condition_guard/20260629_detailfilm_guard_screen64_plan.sh

results/stage5_condition_guard/20260629_detailfilm_guard_full552_plan.json
results/stage5_condition_guard/20260629_detailfilm_guard_full552_plan.sh
```

Each plan contains:

```text
none baseline
global  RMS ratios: 0.15 / 0.25 / 0.35 / 0.50 / 0.75 / 1.00
channel RMS ratios: 0.15 / 0.25 / 0.35 / 0.50 / 0.75 / 1.00
spatial RMS ratios: 0.15 / 0.25 / 0.35 / 0.50 / 0.75 / 1.00
```

Use screen64 first after GPU restart:

```bash
bash results/stage5_condition_guard/20260629_detailfilm_guard_screen64_plan.sh
```

Generated guarded LoRA training plan from the same detail-FiLM anchor:

```text
results/stage5_lora/20260629_detailfilm_lora_guarded_denoiser_tail_decnet_limit64_plan.json
results/stage5_lora/20260629_detailfilm_lora_guarded_denoiser_tail_decnet_limit64_plan.sh
```

This uses:

```text
--condition-residual-rms-guard-weight 0.05
--condition-residual-rms-guard-ratio 0.35
--condition-residual-rms-guard-granularity channel
```

Only promote promising guarded settings to full552. The full552 plan exists for
reproducibility, but should not be run blindly if screen64 shows clear
failures.

## Promotion Criteria

Promote a guard setting only if it improves the current no-guard branch without
turning into a final-output blend:

```text
actual_payload_bpp unchanged
LPIPS and/or DISTS improve over guard_none
PSNR/MS-SSIM do not collapse beyond documented Stage3 guards
condition_l1 does not regress severely, or visual audit explains the tradeoff
per-split behavior is not driven by only Kodak or only CLIC
visual audit shows fewer structure failures, not just smoother images
```

This is still not a Stage 5 external-baseline win unless full552 and official
baseline comparisons show it. It is a mainline stability probe that can be
combined later with:

```text
LoRA / partial backbone adaptation
stronger condition adapter training
tiny counted condition-control stream
diffusion-friendly detail heads
multi-rate BD-curve construction
```

## CPU Verification

GPU remained unavailable:

```text
nvidia-smi: Failed to initialize NVML: Unknown Error
torch.cuda.is_available: False
```

No GPU evaluation was launched.

Passed:

```bash
.venv/bin/python -m py_compile \
  scripts/eval_stage4_cod_lite_adapter.py \
  scripts/sweep_stage5_condition_residual_guard.py \
  scripts/select_stage5_control_candidates.py \
  src/coserdic/models/gencodec_backbone.py
```

Passed:

```bash
PYTHONPATH=/workspace/CoSER-DiC .venv/bin/pytest \
  tests/test_gencodec_backbone.py \
  tests/test_sweep_stage5_condition_residual_guard.py \
  tests/test_select_stage5_control_candidates.py -q
```

Result:

```text
32 passed
```
