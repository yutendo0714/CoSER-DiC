# 043 GPU Preflight for Long Experiments

Date: 2026-06-29

## Decision

All generated long-running GPU experiment shell plans should fail fast when
CUDA/NVML is not healthy.

This is an operational guardrail, not a modeling change.

## Motivation

The current container sometimes stops seeing the GPU. When this happens, CoSER
training/evaluation must not silently continue on CPU or produce misleading
timing/results.

Current observed failure:

```text
nvidia-smi:
  Failed to initialize NVML: Unknown Error

torch:
  torch.cuda.is_available() == false
  torch.cuda.device_count() == 0
```

Per project policy, GPU training/evaluation should stop until the container is
restarted.

## Implementation

Added:

```text
scripts/check_gpu_ready.py
```

It checks:

```text
torch.cuda.is_available()
torch.cuda.device_count() >= min_devices
nvidia-smi exits successfully
```

and exits nonzero when the GPU is not ready.

The following plan generators now put this at the top of generated `.sh` files:

```text
.venv/bin/python scripts/check_gpu_ready.py --min-devices 1
```

Affected generators:

```text
scripts/sweep_stage5_condition_residual_guard.py
scripts/sweep_stage5_lora_backbone.py
scripts/promote_stage5_control_candidates.py
scripts/plan_cod_lite_official_baselines.py
```

Direct-execution sweeps also run the same preflight once before launching eval
commands unless `--skip-gpu-preflight` is explicitly set:

```text
scripts/sweep_stage5_condition_residual_guard.py
scripts/sweep_stage5_counted_control.py
```

Regenerated GPU plans:

```text
results/stage5_condition_guard/20260629_detailfilm_guard_screen64_plan.sh
results/stage5_condition_guard/20260629_detailfilm_guard_full552_plan.sh
results/stage5_lora/20260629_detailfilm_lora_guarded_denoiser_tail_decnet_limit64_plan.sh
results/baselines/cod_lite_official/20260629_cod512_full552_plan.sh
results/baselines/cod_one_step_official/20260629_cod512_full552_plan.sh
```

## Research Impact

This does not change:

```text
actual_payload_bpp
model architecture
training objective
evaluation metric definitions
```

It prevents invalid Stage 5 execution when the GPU is unavailable.

## Verification

CPU tests:

```text
pytest tests/test_check_gpu_ready.py \
  tests/test_sweep_stage5_condition_residual_guard.py \
  tests/test_sweep_stage5_lora_backbone.py \
  tests/test_promote_stage5_control_candidates.py \
  tests/test_plan_cod_lite_official_baselines.py -q
```

Current environment preflight correctly fails because NVML is unavailable.
