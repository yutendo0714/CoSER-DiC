# 20260629_stage4_fast8192_detailcontrol_ft800_b4

Date: 2026-06-29T02:07:42

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage4_fast8192_detailcontrol_ft800_b4 --crop-size 512 --batch-size 4 --num-workers 4 --max-steps 800 --lr 5e-6 --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --detail-control-branch --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailaware_ft600_fast8192_ft1500_b4.pt --init-nonstrict --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 1.0 --condition-cosine-weight 0.20 --condition-channel-stats-weight 0.15 --condition-highfreq-weight 0.03 --image-l1-weight 0.25 --lpips-weight 0.03 --ms-ssim-weight 0.02 --stage3-l1-guard-weight 0.20 --stage3-mse-guard-weight 0.05 --grad-clip-norm 1.0 --save-sample-every 200 --wandb-mode offline
```

## Summary

- loss_mean: 0.5124762801080942
- condition_l1_mean: 0.3957068744674325
- condition_cosine_loss_mean: 0.2224373511970043
- condition_channel_stats_loss_mean: 0.250469677541405
- condition_highfreq_loss_mean: 0.0560844686254859
- base_condition_l1_mean: 0.5169031979143619
- condition_l1_delta_vs_base_mean: -0.12119632344692946
- condition_cosine_mean: 0.7775626488029956
- pred_condition_std_mean: 0.7362911486625672
- target_condition_std_mean: 0.8089291990548372
- pred_condition_highfreq_mean: 0.22305508941411972
- target_condition_highfreq_mean: 0.24535947298631072
- image_l1_mean: 0.05807680185418576
- lpips_mean: 0.40464974127709863
- dists_loss_mean: 0.0
- ms_ssim_loss_mean: 0.27690666176378725
- stage3_l1_guard_mean: 0.0038293624747893773
- stage3_mse_guard_mean: 0.0013250067867556937
- stage4_psnr_mean: 21.23968712806702
- stage3_psnr_mean: 21.972161614894866
- condition_residual_l1_mean: 0.33765558548271657
- condition_delta_raw_l1_mean: 0.5266046470031143
- semantic_latent_drop_fraction_mean: 0.0
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailcontrol_ft800_b4.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailcontrol_ft800_b4/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailcontrol_ft800_b4`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_020410-ljbuny7y`
