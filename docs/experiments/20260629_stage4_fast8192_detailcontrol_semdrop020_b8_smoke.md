# 20260629_stage4_fast8192_detailcontrol_semdrop020_b8_smoke

Date: 2026-06-29T02:12:38

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage4_fast8192_detailcontrol_semdrop020_b8_smoke --crop-size 512 --batch-size 8 --num-workers 4 --max-steps 2 --lr 5e-6 --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --detail-control-branch --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailcontrol_ft800_b4.pt --semantic-latent-dropout-prob 0.20 --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 1.0 --condition-cosine-weight 0.20 --condition-channel-stats-weight 0.15 --condition-highfreq-weight 0.03 --image-l1-weight 0.25 --lpips-weight 0.03 --ms-ssim-weight 0.02 --stage3-l1-guard-weight 0.20 --stage3-mse-guard-weight 0.05 --grad-clip-norm 1.0 --save-sample-every 0 --wandb-mode offline
```

## Summary

- loss_mean: 0.5286515951156616
- condition_l1_mean: 0.4030090570449829
- condition_cosine_loss_mean: 0.23264247179031372
- condition_channel_stats_loss_mean: 0.2568703144788742
- condition_highfreq_loss_mean: 0.054647473618388176
- base_condition_l1_mean: 0.5324058532714844
- condition_l1_delta_vs_base_mean: -0.12939679622650146
- condition_cosine_mean: 0.7673575282096863
- pred_condition_std_mean: 0.7233157157897949
- target_condition_std_mean: 0.8076907694339752
- pred_condition_highfreq_mean: 0.22616221010684967
- target_condition_highfreq_mean: 0.24189544469118118
- image_l1_mean: 0.07417457923293114
- lpips_mean: 0.4126930236816406
- dists_loss_mean: 0.0
- ms_ssim_loss_mean: 0.32080385088920593
- stage3_l1_guard_mean: 0.007517632097005844
- stage3_mse_guard_mean: 0.002000153064727783
- stage4_psnr_mean: 19.843597412109375
- stage3_psnr_mean: 20.795090675354004
- condition_residual_l1_mean: 0.3446355015039444
- condition_delta_raw_l1_mean: 0.5449004471302032
- semantic_latent_drop_fraction_mean: 0.125
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailcontrol_semdrop020_b8_smoke.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailcontrol_semdrop020_b8_smoke/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailcontrol_semdrop020_b8_smoke`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_021234-0m5v97x1`
