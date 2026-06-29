# Stage 4 Detail-FiLM Condition Adapter

Date: 2026-06-29

## Decision

Promote the Detail-FiLM adapter change as mainline Stage 4 infrastructure and
as the current best bpp0.0156 no-extra-bit adapter variant, but do not call it a
Stage 5 or external-baseline result.

The improvement is small but directionally correct:

```text
decoded detail payload now modulates the condition-space fusion path
full552 improves PSNR, LPIPS, DISTS, and condition L1 versus the previous
semantic-dropout detail-control adapter
MS-SSIM slightly regresses
```

This stays inside the CoSER-DiC core method:

```text
decoded semantic/detail actual bitstream
  -> CoSER condition adapter with detail FiLM modulation
  -> frozen official CoD-Lite rate checkpoint
  -> Stage 4 reconstruction
```

No RGB output blending is used as the method. No image-specific side
information is transmitted, so `actual_payload_bpp / paper_bpp` remains the
Stage 3 semantic/detail payload.

## Implementation

Added a detail FiLM branch to the pyramid condition adapter:

```text
src/coserdic/models/gencodec_backbone.py

CoSERToCoDLiteConditionPyramidAdapterConfig.detail_film_modulation
```

Training CLI:

```text
scripts/train_stage4_cod_lite_adapter.py

--detail-film-modulation
```

Mechanism:

```text
detail_features = detail_blocks(detail_proj(decoded_detail_context))
gamma, beta = zero_init_conv(detail_features).chunk(2)
fused = fused * (1 + gamma) + beta
condition_delta = out(fusion_blocks(fused)) + optional_detail_condition_delta
```

The FiLM branch is zero-initialized. Existing checkpoints can be loaded with
`--init-nonstrict`, and the initial behavior is preserved except for learned
updates during continuation training.

Added tests:

```text
tests/test_gencodec_backbone.py

test_coser_to_cod_lite_pyramid_adapter_detail_film_modulation_zero_init
test_coser_to_cod_lite_pyramid_adapter_detail_film_requires_detail_context
```

## Training Probe

Run:

```text
20260629_stage4_fast8192_detailfilm_ft800_b8
```

Init:

```text
checkpoints/stage4_cod_lite_adapter/
20260629_stage4_fast8192_detailcontrol_semdrop020_ft800_b8.pt
```

Training data:

```text
results/bitstreams/stage3_training_cache_fast/
20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4
```

Backbone:

```text
official CoD-Lite bpp0.0156 checkpoint
```

Training settings:

```text
batch_size: 8
steps: 800
lr: 5e-6
semantic_latent_dropout_prob: 0.20
condition_residual_scale: 0.85
condition_residual_tanh: true
detail_control_branch: true
detail_film_modulation: true
```

Training summary:

| metric | value |
| --- | ---: |
| condition_l1_mean | 0.3950 |
| image_l1_mean | 0.0594 |
| LPIPS mean | 0.4046 |
| MS-SSIM loss mean | 0.2800 |
| stage4_psnr_mean | 21.0780 |
| condition_residual_l1_mean | 0.3364 |

## Limit64 Evaluation

Strict 512 first-64 subset from the full552 feature cache:

| checkpoint | PSNR | MS-SSIM | LPIPS | DISTS | condition_l1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| semdrop detail-control raw | 21.6120 | 0.7393 | 0.4098 | 0.2965 | 0.4064 |
| detail-FiLM | 21.6393 | 0.7389 | 0.4091 | 0.2960 | 0.4065 |

Detail ablation for the Detail-FiLM checkpoint:

| detail context | PSNR | MS-SSIM | LPIPS | DISTS | condition_l1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| normal | 21.6393 | 0.7389 | 0.4091 | 0.2960 | 0.4065 |
| zero | 21.6701 | 0.7388 | 0.4122 | 0.2994 | 0.4067 |
| shuffle | 21.6525 | 0.7386 | 0.4108 | 0.2981 | 0.4067 |

Interpretation:

```text
normal detail improves LPIPS/DISTS relative to zero/shuffle
detail FiLM uses detail mainly for perceptual realism, with a small fidelity cost
```

## Full552 Evaluation

Strict CoD-style 512 protocol:

```text
Kodak24 + CLIC2020 test428 + DIV2K val100
actual_payload_bpp: 0.013999
```

Overall comparison:

| checkpoint | PSNR | MS-SSIM | LPIPS | DISTS | condition_l1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| semdrop detail-control raw | 21.2325 | 0.7154 | 0.4315 | 0.2988 | 0.4069 |
| detail-FiLM | 21.2427 | 0.7150 | 0.4304 | 0.2982 | 0.4060 |

Delta:

```text
PSNR:     +0.0102 dB
MS-SSIM:  -0.0004
LPIPS:    -0.0011
DISTS:    -0.0006
cond L1:  -0.0010
```

Split notes:

```text
CLIC2020 test:
  PSNR, LPIPS, DISTS, condition L1 improve; MS-SSIM slightly regresses

DIV2K val:
  PSNR, LPIPS, condition L1 improve; DISTS is essentially flat/slightly worse

Kodak:
  PSNR, LPIPS improve; DISTS and condition L1 slightly regress
```

This is not a large win, but it is better aligned with the mainline than
post-hoc output blending because the change lives in the condition adapter.

## Blocked Follow-up

The planned bpp0.0312 Detail-FiLM continuation was started but stopped because
the container lost GPU access:

```text
RuntimeError: CUDA is not visible
NVML: Failed to initialize NVML: Unknown Error
torch.cuda.is_available = False
```

Per project policy, no further GPU experiment should be launched until the
container is restarted.

After restart, the next command to try is:

```bash
.venv/bin/python scripts/train_stage4_cod_lite_adapter.py \
  --cod-lite-checkpoint external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0312.pt \
  --cod-lite-config external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0312.yaml \
  --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl \
  --run-name 20260629_stage4_bpp0312_fast8192_detailcontrol_detailfilm_ft800_b8 \
  --crop-size 512 --batch-size 8 --num-workers 4 --max-steps 800 --lr 5e-6 \
  --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes \
  --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 \
  --num-detail-blocks 3 --num-fusion-blocks 4 \
  --detail-control-branch --detail-film-modulation \
  --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_detailaware_adapter_bpp0312_transfer_ft600_b2.pt \
  --init-nonstrict --semantic-latent-dropout-prob 0.20 \
  --condition-residual-scale 0.85 --condition-residual-tanh \
  --condition-l1-weight 1.0 --condition-cosine-weight 0.25 \
  --condition-channel-stats-weight 0.20 --condition-highfreq-weight 0.05 \
  --image-l1-weight 0.08 --lpips-weight 0.03 --ms-ssim-weight 0.08 \
  --stage3-l1-guard-weight 0.60 --stage3-mse-guard-weight 1.20 \
  --grad-clip-norm 1.0 --save-sample-every 200 --wandb-mode offline
```

Then evaluate:

```bash
.venv/bin/python scripts/eval_stage4_cod_lite_adapter.py \
  --checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_bpp0312_fast8192_detailcontrol_detailfilm_ft800_b8.pt \
  --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl \
  --per-image-metrics results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/per_image_metrics.jsonl \
  --run-name 20260629_stage4_bpp0312_fast8192_detailcontrol_detailfilm_ft800_b8_limit64_eval \
  --crop-size 512 --limit 64 --batch-size 4 --num-workers 4 \
  --blend-alpha 1.0 --wandb-mode offline
```

Promote to full552 only if limit64 improves the bpp0.0312 transfer anchor in
at least LPIPS or DISTS without a severe PSNR/MS-SSIM regression.

## Research Judgment

Promote:

```text
Detail-FiLM as a better mainline adapter mechanism
20260629_stage4_fast8192_detailfilm_ft800_b8 as the current bpp0.0156
no-extra-bit adapter variant
```

Do not promote:

```text
external-baseline superiority
Stage 5 completion
RGB output blending or fixed alpha as the method
```

Next research should continue along:

```text
rate-specific bpp0.0312 Detail-FiLM transfer
condition-channel/group-specific detail targets
joint adapter+condition-gate only after stronger base adapter
larger clean train cache
eventual tiny counted control stream only after decoder-side adapter saturates
```
