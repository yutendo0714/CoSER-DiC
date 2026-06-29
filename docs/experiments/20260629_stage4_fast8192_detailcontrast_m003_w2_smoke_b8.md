# 20260629_stage4_fast8192_detailcontrast_m003_w2_smoke_b8

Date: 2026-06-29T02:25:02

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage4_fast8192_detailcontrast_m003_w2_smoke_b8 --crop-size 512 --batch-size 8 --num-workers 4 --max-steps 2 --lr 5e-6 --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --detail-control-branch --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailcontrol_ft800_b4.pt --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 1.0 --condition-cosine-weight 0.20 --condition-channel-stats-weight 0.15 --condition-highfreq-weight 0.03 --detail-contrast-weight 2.0 --detail-contrast-margin 0.003 --image-l1-weight 0.25 --lpips-weight 0.03 --ms-ssim-weight 0.02 --stage3-l1-guard-weight 0.20 --stage3-mse-guard-weight 0.05 --grad-clip-norm 1.0 --save-sample-every 0 --wandb-mode offline
```

## Summary

- loss_mean: 0.5253330171108246
- condition_l1_mean: 0.39960214495658875
- condition_cosine_loss_mean: 0.22883743047714233
- condition_channel_stats_loss_mean: 0.24590115994215012
- condition_highfreq_loss_mean: 0.05553255043923855
- detail_contrast_loss_mean: 0.0019915341399610043
- detail_condition_l1_zero_mean: 0.4006106108427048
- detail_condition_l1_gap_mean: 0.0010084658861160278
- base_condition_l1_mean: 0.5324058532714844
- condition_l1_delta_vs_base_mean: -0.13280370831489563
- condition_cosine_mean: 0.7711625695228577
- pred_condition_std_mean: 0.7321608066558838
- target_condition_std_mean: 0.8076907694339752
- pred_condition_highfreq_mean: 0.22293413430452347
- target_condition_highfreq_mean: 0.24189544469118118
- image_l1_mean: 0.07121777161955833
- lpips_mean: 0.4087556302547455
- dists_loss_mean: 0.0
- ms_ssim_loss_mean: 0.3183136582374573
- stage3_l1_guard_mean: 0.004560828907415271
- stage3_mse_guard_mean: 0.0016718110418878496
- stage4_psnr_mean: 20.000627517700195
- stage3_psnr_mean: 20.795090675354004
- condition_residual_l1_mean: 0.3427521288394928
- condition_delta_raw_l1_mean: 0.5408501029014587
- semantic_latent_drop_fraction_mean: 0.0
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailcontrast_m003_w2_smoke_b8.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailcontrast_m003_w2_smoke_b8/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailcontrast_m003_w2_smoke_b8`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_022458-ckjt1bvr`
