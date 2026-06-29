# Stage 5 Partial CoD-Lite Backbone Finetune Path

Date: 2026-06-29

## Decision

Add a mainline Stage 5 path for low-LR partial adaptation of the official
CoD-Lite one-step backbone, driven only by decoded CoSER semantic/detail
features and the CoSER-owned condition adapter.

This is not a post-hoc RGB fix. It is a decoder-conditioning method:

```text
decoded CoSER semantic/detail actual payload
  -> Detail-FiLM / future condition adapter
  -> CoSER-predicted CoD-Lite condition
  -> partially adapted official CoD-Lite decoder backbone
  -> reconstruction
```

The backbone adaptation is stored as fixed model state. It is not image-specific
side information and must not be counted in `actual_payload_bpp`. If a future
method transmits per-image controls, maps, masks, prompts, seeds, or condition
tokens, those bits must be entropy-coded and counted.

## Why This Is Mainline

Current adapter-only Stage 4 has useful signal but is far from official
CoD-Lite perceptual quality:

```text
current CoSER Kodak LPIPS/DISTS:
  LPIPS best checked: about 0.4678
  DISTS best checked: about 0.3126

official CoD-Lite Kodak512 worst stored point:
  LPIPS: 0.3390
  DISTS: 0.2089
```

This means small adapter cleanups cannot plausibly become Stage 5 by
themselves. The next high-value move is to let the pretrained one-step decoder
adapt slightly to CoSER's condition distribution while keeping:

```text
CoSER semantic/detail streams as the only transmitted payload
official CoD-Lite as a referenced pretrained diffusion backbone
CoSER-owned adapter/control as the proposed codec mechanism
actual_payload_bpp unchanged
```

## Official Implementation Reference

The local official CoD-Lite implementation supports this direction:

```text
external/repos/GenCodec/CoD_Lite/finetuned_one_step_codec/main_stage1.py
  uses LoRA over Linear/Conv modules
  excludes y_embedder / conv2 from LoRA targets
  saves y_embedder and conv2 as special modules

external/repos/GenCodec/CoD_Lite/finetuned_one_step_codec/main_stage2.py
  freezes conditioner / VAE / teacher denoisers
  can freeze y_embedder encoder and bottleneck
  trains denoiser-side modules in the one-step codec
```

CoSER-DiC should not copy this training pipeline wholesale. Instead, it should
borrow the architectural lesson:

```text
the decoder/denoiser can be adapted
the condition codec encoder should be treated carefully
LoRA is a reasonable parallel path to low-LR partial unfreeze
```

## Implementation

Added name-based partial unfreeze helpers:

```text
src/coserdic/models/gencodec_backbone.py

compile_parameter_name_patterns
set_trainable_parameters_by_name
LoRALinear
LoRAConv2d
apply_lora_adapters_by_name
apply_lora_adapters_from_config
lora_parameter_names
named_parameter_state
load_named_parameter_state
```

Training CLI:

```text
scripts/train_stage4_cod_lite_adapter.py

--backbone-train-pattern REGEX
--backbone-lr FLOAT
--backbone-weight-decay FLOAT
--backbone-lora-pattern REGEX
--backbone-lora-rank INT
--backbone-lora-alpha FLOAT
--backbone-lora-lr FLOAT
--backbone-lora-target linear|conv2d
```

Checkpoint payload now stores:

```text
backbone_train_pattern
backbone_lr
backbone_weight_decay
backbone_lora_config
backbone_trainable_param_names
backbone_trainable_state
```

The state contains only selected trainable tensors and LoRA tensors, not a full
copy of the official checkpoint.

Evaluation / analysis scripts load this state before decoding:

```text
scripts/eval_stage4_cod_lite_adapter.py
scripts/analyze_stage4_cod_lite_condition_stats.py
scripts/train_stage4_cod_lite_condition_gate.py
scripts/train_stage4_cod_lite_gate.py
scripts/fit_stage5_condition_control_basis.py
```

The train script now fails if a requested regex matches no backbone parameters.
This prevents accidentally running an adapter-only experiment while believing
the decoder was partially adapted.

Added a restart-only inspection helper:

```text
scripts/inspect_cod_lite_backbone_params.py
```

Use it after GPU recovery to record the post-`eval()` CoD-Lite parameter names
and the exact regex match set before launching long training.

## First Target Pattern

Start conservative:

```text
final_layer|y_embedder_x|blocks\.(2[0-9]|3[0-9])|dec_net
```

Rationale:

```text
final_layer:
  directly controls final pixel prediction

y_embedder_x:
  maps condition features into the image reconstruction stream

late blocks / dec_net:
  adapt the decoder-side rendering path after condition features are formed

avoid y_embedder encoder / bottleneck at first:
  target native conditions should remain a stable teacher signal
```

If memory or instability appears, shrink to:

```text
final_layer|y_embedder_x
```

If the conservative run is stable but weak, expand to:

```text
final_layer|y_embedder_x|blocks\.(1[6-9]|2[0-9]|3[0-9])|dec_net
```

LoRA should be tested only after this low-LR path gives a clean baseline.

## First GPU-Restart Command

GPU is currently unavailable:

```text
nvidia-smi: Failed to initialize NVML: Unknown Error
torch.cuda.is_available: False
```

Per project policy, do not launch this until the container is restarted.

After restart, run:

```bash
.venv/bin/python scripts/inspect_cod_lite_backbone_params.py \
  --cod-lite-checkpoint external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0312.pt \
  --cod-lite-config external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0312.yaml \
  --pattern 'final_layer|y_embedder_x|dec_net' \
  --output-json results/stage5_inspect/20260629_cod_lite_bpp0312_partial_final_yx_decnet_params.json
```

If the matched tensor count is reasonable, launch:

```bash
.venv/bin/python scripts/train_stage4_cod_lite_adapter.py \
  --cod-lite-checkpoint external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0312.pt \
  --cod-lite-config external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0312.yaml \
  --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl \
  --run-name 20260629_stage5_bpp0312_detailfilm_partial_final_yx_decnet_ft1000_b4ga2 \
  --crop-size 512 --batch-size 4 --grad-accum-steps 2 --num-workers 4 --max-steps 1000 \
  --lr 3e-6 --backbone-lr 5e-7 --backbone-weight-decay 0.0 \
  --backbone-train-pattern 'final_layer|y_embedder_x|dec_net' \
  --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes \
  --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 \
  --num-detail-blocks 3 --num-fusion-blocks 4 \
  --detail-control-branch --detail-film-modulation \
  --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_detailaware_adapter_bpp0312_transfer_ft600_b2.pt \
  --init-nonstrict --semantic-latent-dropout-prob 0.20 \
  --condition-residual-scale 0.85 --condition-residual-tanh \
  --condition-l1-weight 1.0 --condition-cosine-weight 0.20 \
  --condition-channel-stats-weight 0.15 --condition-highfreq-weight 0.03 \
  --image-l1-weight 0.10 --lpips-weight 0.08 --dists-weight 0.08 --ms-ssim-weight 0.04 \
  --stage3-l1-guard-weight 0.50 --stage3-mse-guard-weight 1.00 \
  --grad-clip-norm 1.0 --save-sample-every 250 --wandb-mode offline
```

Then evaluate limit64 before full552:

```bash
.venv/bin/python scripts/eval_stage4_cod_lite_adapter.py \
  --checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage5_bpp0312_detailfilm_partial_final_yx_decnet_ft1000_b4ga2.pt \
  --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl \
  --per-image-metrics results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/per_image_metrics.jsonl \
  --run-name 20260629_stage5_bpp0312_detailfilm_partial_final_yx_decnet_ft1000_b4ga2_limit64_eval \
  --crop-size 512 --limit 64 --batch-size 4 --num-workers 4 \
  --blend-alpha 1.0 --wandb-mode offline
```

Promote to full552 only if the limit64 result improves LPIPS or DISTS over the
bpp0.0312 transfer anchor without severe PSNR/MS-SSIM collapse.

## Promotion Rule

This path is promoted only if it improves final decoded images, not merely
condition L1:

```text
actual_payload_bpp unchanged
roundtrip failure_count = 0
no reference leakage at inference
LPIPS / DISTS improve over current Stage 4 anchors
Kodak LPIPS moves toward or below 0.3390
Kodak DISTS moves toward or below 0.2089
visual structure is audited
BD-rate tooling can eventually compute overlap against official CoD-Lite
```

Low-LR partial unfreeze and LoRA should now be treated as parallel Stage 5
backbone-adaptation tracks. If they saturate, next mainline steps are:

```text
channel/group-specific condition residual targets
diffusion-friendly detail head from the same transmitted detail payload
tiny counted control stream, only if deterministic decoder-side control saturates
CoD heavy backbone parallel track
```

The first tiny counted control-stream path is now implemented and documented in:

```text
docs/research/design_decisions/039_stage5_tiny_counted_condition_control_stream.md
docs/research/design_decisions/040_stage5_lora_backbone_adaptation_path.md
```
