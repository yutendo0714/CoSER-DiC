# 040 Stage 5 LoRA Backbone Adaptation Path

Date: 2026-06-29

## Decision

Add a Stage 5 mainline path for low-rank adaptation of the official CoD-Lite
decoder backbone.

This is not a post-hoc RGB refiner and not a replacement with the native
CoD-Lite bitstream. The transmitted payload remains:

```text
CoSER semantic VQ stream
+ CoSER detail residual stream
+ optional counted control stream
```

The LoRA weights are fixed decoder-side model state, like adapter weights or a
published checkpoint. They are not counted in `actual_payload_bpp`.

## Motivation

Adapter-only Stage 4 improved the internal CoSER substrate but still trails the
official CoD-Lite perceptual curve badly. Full backbone fine-tuning is risky
because it can destroy the pretrained diffusion prior. Name-based partial
unfreeze is useful, but it may move too many parameters and is harder to make
stable.

LoRA is the safer next mainline move:

```text
decoded CoSER semantic/detail condition
  -> CoSER condition adapter
  -> frozen CoD-Lite backbone + small low-rank residual updates
  -> decoded image
```

This gives CoSER a way to adapt the official generative prior to CoSER
condition tensors without changing the bitstream definition.

## Implementation

Added:

```text
src/coserdic/models/gencodec_backbone.py
  LoRALinear
  LoRAConv2d
  apply_lora_adapters_by_name
  apply_lora_adapters_from_config
  lora_parameter_names

scripts/train_stage4_cod_lite_adapter.py
  --backbone-lora-pattern
  --backbone-lora-plan
  --backbone-lora-preset
  --backbone-lora-rank
  --backbone-lora-alpha
  --backbone-lora-lr
  --backbone-lora-weight-decay
  --backbone-lora-target

scripts/inspect_cod_lite_backbone_params.py
  CPU-capable parameter/module inspection
  LoRA candidate module listing
  rank-specific LoRA parameter count estimates

scripts/plan_stage5_lora_targets.py
  denoiser-only / dec_net / y_decoder LoRA target plans
  exact regex generation for train_stage4_cod_lite_adapter.py

scripts/sweep_stage5_lora_backbone.py
  train/eval command plans from LoRA target presets
  limit64 summary paths for candidate selection
```

Checkpoints now save:

```text
backbone_lora_config
backbone_trainable_state
```

where `backbone_trainable_state` contains the LoRA tensors and any explicitly
unfrozen backbone tensors. Evaluation, condition-stat analysis, gate training,
and Stage 5 control-basis fitting reconstruct LoRA modules from
`backbone_lora_config` before loading `backbone_trainable_state`.

## Evaluation Rule

LoRA is allowed only as fixed decoder model state.

Count in `actual_payload_bpp`:

```text
semantic payload
detail payload
optional per-image counted control stream
```

Do not count:

```text
fixed LoRA checkpoint tensors
fixed adapter tensors
fixed CoD-Lite pretrained backbone
```

Do count any future per-image LoRA/control update if such a design is ever
introduced.

## GPU-Restart Plan

First inspect parameter names for the chosen official checkpoint:

```bash
.venv/bin/python scripts/inspect_cod_lite_backbone_params.py \
  --device cpu \
  --cod-lite-checkpoint external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0312.pt \
  --cod-lite-config external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0312.yaml \
  --pattern 'final|dec|decoder|output|attn|mlp|proj|ffn|block' \
  --lora-rank 4 \
  --output-json results/stage5_inspect/20260629_cod_lite_bpp0312_lora_candidate_params_cpu.json
```

This has already been run on CPU because CUDA is currently unavailable.

Observed bpp0.0312 rank-4 candidate scale:

```text
total CoD-Lite params:
  80,153,283

broad matched LoRA candidate modules:
  199 modules
  1,052,044 rank-4 LoRA params
```

Plan target presets from the inspection JSON:

```bash
.venv/bin/python scripts/plan_stage5_lora_targets.py \
  --inspect-json results/stage5_inspect/20260629_cod_lite_bpp0312_lora_candidate_params_cpu.json \
  --preset denoiser_tail \
  --preset dec_net \
  --preset denoiser_mlp \
  --preset denoiser_all \
  --preset y_decoder \
  --last-blocks 6 \
  --output-json results/stage5_inspect/20260629_cod_lite_bpp0312_lora_target_plan_rank4.json
```

Generated first-probe candidates:

```text
denoiser_tail:
  30 modules
  147,456 rank-4 LoRA params
  safest first performance probe; avoids y_embedder condition codec

dec_net:
  11 modules
  37,632 rank-4 LoRA params
  cheapest stability probe

denoiser_mlp:
  42 modules
  322,560 rank-4 LoRA params

denoiser_all:
  105 modules
  516,096 rank-4 LoRA params

y_decoder:
  16 modules
  138,240 rank-4 LoRA params
  riskier because it touches condition-codec decoder
```

Then run a conservative LoRA continuation from the current strongest
detail-FiLM / bpp0.0312 adapter anchor. Prefer `denoiser_tail` first:

```bash
.venv/bin/python scripts/sweep_stage5_lora_backbone.py \
  --lora-plan results/stage5_inspect/20260629_cod_lite_bpp0312_lora_target_plan_rank4.json \
  --preset denoiser_tail \
  --preset dec_net \
  --run-prefix 20260629_stage5_bpp0312_lora \
  --train-manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl \
  --eval-manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl \
  --eval-per-image-metrics results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/per_image_metrics.jsonl \
  --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_detailaware_adapter_bpp0312_transfer_ft600_b2.pt \
  --max-steps 1000 \
  --output-json results/stage5_lora/20260629_bpp0312_lora_denoiser_tail_decnet_limit64_plan.json \
  --output-sh results/stage5_lora/20260629_bpp0312_lora_denoiser_tail_decnet_limit64_plan.sh
```

The generated shell file contains train + limit64 eval commands for:

```text
20260629_stage5_bpp0312_lora_denoiser_tail_r4_lr2em05_ft1000
20260629_stage5_bpp0312_lora_dec_net_r4_lr2em05_ft1000
```

Evaluate limit64 before any full552 promotion:

```bash
bash results/stage5_lora/20260629_bpp0312_lora_denoiser_tail_decnet_limit64_plan.sh
```

After the detail-FiLM anchor became the strongest no-extra-bit Stage 4 branch,
an updated guarded LoRA continuation plan was generated from:

```text
checkpoints/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailfilm_ft800_b8.pt
```

Generated:

```text
results/stage5_lora/20260629_detailfilm_lora_guarded_denoiser_tail_decnet_limit64_plan.json
results/stage5_lora/20260629_detailfilm_lora_guarded_denoiser_tail_decnet_limit64_plan.sh
```

This plan adds training-time condition residual RMS regularization:

```text
--condition-residual-rms-guard-weight 0.05
--condition-residual-rms-guard-ratio 0.35
--condition-residual-rms-guard-granularity channel
```

Run this newer plan first after GPU restart if prioritizing the current
detail-FiLM anchor:

```bash
bash results/stage5_lora/20260629_detailfilm_lora_guarded_denoiser_tail_decnet_limit64_plan.sh
```

Promotion criteria:

```text
actual_payload_bpp unchanged unless counted control is also enabled
LPIPS/DISTS improve over the current bpp0.0312 raw anchor
PSNR/MS-SSIM do not collapse beyond documented guards
condition_l1 improves or visual audit explains why it does not
semantic/detail ablations still show CoSER bitstream dependence
```

## Expected Outcome

Short term:

```text
stabilize CoSER-conditioned CoD-Lite decoding
close more of the LPIPS/DISTS gap without paying extra payload bits
```

Medium term:

```text
combine LoRA backbone adaptation with the tiny counted control stream
build multi-rate curves and compute protocol-matched BD-rate
```

This is still not a Stage 5 win until full552 and protocol-matched baseline
curves show it.
