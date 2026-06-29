# 20260629_stage4_stats_hf_condition_gate_channel_balanced_smoke_b8

Date: 2026-06-29T11:39:53

## Command

```bash
scripts/train_stage4_cod_lite_condition_gate.py --adapter-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailfilm_stats_hf_ft1200_b12.pt --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage4_stats_hf_condition_gate_channel_balanced_smoke_b8 --crop-size 512 --batch-size 8 --grad-accum-steps 1 --num-workers 4 --max-steps 10 --lr 1e-4 --hidden-channels 192 --num-blocks 3 --init-gate 0.85 --channel-gate --condition-l1-weight 0.02 --condition-regression-guard-weight 0.05 --image-l1-weight 0.8 --lpips-weight 0.08 --dists-weight 0.15 --ms-ssim-weight 0.2 --stage3-l1-guard-weight 1.0 --stage3-mse-guard-weight 2.0 --wandb-mode offline
```

## Summary

- loss_mean: 0.19835179001092912
- image_l1_mean: 0.06200581565499306
- lpips_mean: 0.41772923469543455
- dists_loss_mean: 0.27436048090457915
- ms_ssim_loss_mean: 0.2913422405719757
- condition_l1_mean: 0.4013438284397125
- ungated_condition_l1_mean: 0.3986842125654221
- condition_regression_guard_mean: 0.0027350395917892455
- stage3_l1_guard_mean: 0.0043972636805847285
- stage3_mse_guard_mean: 0.0016726928646676243
- gate_mean_loss_mean: 0.0
- stage4_psnr_mean: 20.526175498962402
- stage3_psnr_mean: 21.327783012390135
- stage4_ms_ssim_mean: 0.7086577594280243
- stage3_ms_ssim_mean: 0.7283902406692505
- gate_mean_mean: 0.8516057848930358
- gate_std_mean: 0.002055903814471094
- gate_min_mean: 0.841015625
- gate_max_mean: 0.85859375
- gate_mean_over_steps: 0.8516057848930358
- gate_mean_std_over_steps: 0.0006184522062540054
- payload_policy: deterministic decoder-side condition gate from decoded CoSER tensors and fixed model weights; no transmitted side information and no RGB output blending

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_condition_gate/20260629_stage4_stats_hf_condition_gate_channel_balanced_smoke_b8.pt`
- summary: `results/stage4_cod_lite_condition_gate/20260629_stage4_stats_hf_condition_gate_channel_balanced_smoke_b8/summary.json`
- results_dir: `results/stage4_cod_lite_condition_gate/20260629_stage4_stats_hf_condition_gate_channel_balanced_smoke_b8`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_113945-1l0kgwmh`
