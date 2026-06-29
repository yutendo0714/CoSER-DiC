# 20260629_stage4_fast8192_detailhfres_w05_smoke_b8

Date: 2026-06-29T02:38:04

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage4_fast8192_detailhfres_w05_smoke_b8 --crop-size 512 --batch-size 8 --num-workers 2 --max-steps 20 --lr 5e-5 --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --detail-control-branch --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailcontrol_ft800_b4.pt --init-nonstrict --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 1.0 --condition-cosine-weight 0.2 --condition-channel-stats-weight 0.15 --condition-highfreq-weight 0.03 --detail-highfreq-residual-weight 0.5 --detail-highfreq-kernel-size 5 --image-l1-weight 0.25 --lpips-weight 0.03 --ms-ssim-weight 0.02 --stage3-l1-guard-weight 0.2 --stage3-mse-guard-weight 0.05 --grad-clip-norm 1.0 --wandb-mode offline --save-sample-every 0
```

## Summary

- loss_mean: 0.6656182944774628
- condition_l1_mean: 0.3992148533463478
- condition_cosine_loss_mean: 0.2315769761800766
- condition_channel_stats_loss_mean: 0.2628835432231426
- condition_highfreq_loss_mean: 0.05816854201257229
- detail_contrast_loss_mean: 0.0
- detail_condition_l1_zero_mean: 0.0
- detail_condition_l1_gap_mean: 0.0
- detail_highfreq_residual_loss_mean: 0.2913340449333191
- detail_highfreq_residual_pred_l1_mean: 0.009571524104103446
- detail_highfreq_residual_target_l1_mean: 0.2911325335502625
- base_condition_l1_mean: 0.5178496524691582
- condition_l1_delta_vs_base_mean: -0.11863479912281036
- condition_cosine_mean: 0.7684230238199234
- pred_condition_std_mean: 0.7273842871189118
- target_condition_std_mean: 0.803721821308136
- pred_condition_highfreq_mean: 0.22586798742413522
- target_condition_highfreq_mean: 0.24704426527023315
- image_l1_mean: 0.05796150639653206
- lpips_mean: 0.4080806881189346
- dists_loss_mean: 0.0
- ms_ssim_loss_mean: 0.27933312952518463
- stage3_l1_guard_mean: 0.004272266756743193
- stage3_mse_guard_mean: 0.0013906033418606967
- stage4_psnr_mean: 21.27000160217285
- stage3_psnr_mean: 22.055769157409667
- condition_residual_l1_mean: 0.3377049744129181
- condition_delta_raw_l1_mean: 0.5271793201565742
- semantic_latent_drop_fraction_mean: 0.0
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailhfres_w05_smoke_b8.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailhfres_w05_smoke_b8/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailhfres_w05_smoke_b8`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_023751-8hti7wow`
