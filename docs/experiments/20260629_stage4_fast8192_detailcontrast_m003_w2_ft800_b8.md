# 20260629_stage4_fast8192_detailcontrast_m003_w2_ft800_b8

Date: 2026-06-29T02:32:50

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage4_fast8192_detailcontrast_m003_w2_ft800_b8 --crop-size 512 --batch-size 8 --num-workers 4 --max-steps 800 --lr 5e-6 --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --detail-control-branch --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailcontrol_ft800_b4.pt --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 1.0 --condition-cosine-weight 0.20 --condition-channel-stats-weight 0.15 --condition-highfreq-weight 0.03 --detail-contrast-weight 2.0 --detail-contrast-margin 0.003 --image-l1-weight 0.25 --lpips-weight 0.03 --ms-ssim-weight 0.02 --stage3-l1-guard-weight 0.20 --stage3-mse-guard-weight 0.05 --grad-clip-norm 1.0 --save-sample-every 200 --wandb-mode offline
```

## Summary

- loss_mean: 0.5127264794707298
- condition_l1_mean: 0.3946757495030761
- condition_cosine_loss_mean: 0.22123084969818593
- condition_channel_stats_loss_mean: 0.25301300166174767
- condition_highfreq_loss_mean: 0.05604989121202379
- detail_contrast_loss_mean: 0.0006001184717752039
- detail_condition_l1_zero_mean: 0.39916837118566034
- detail_condition_l1_gap_mean: 0.0044926216825842856
- base_condition_l1_mean: 0.5168800438940525
- condition_l1_delta_vs_base_mean: -0.12220429439097642
- condition_cosine_mean: 0.778769150301814
- pred_condition_std_mean: 0.7362291537970305
- target_condition_std_mean: 0.8098107255250215
- pred_condition_highfreq_mean: 0.22278092881664635
- target_condition_highfreq_mean: 0.24497095528990032
- image_l1_mean: 0.05780140582937747
- lpips_mean: 0.4052943690866232
- dists_loss_mean: 0.0
- ms_ssim_loss_mean: 0.2766861742734909
- stage3_l1_guard_mean: 0.003802979609463364
- stage3_mse_guard_mean: 0.0013474191757995868
- stage4_psnr_mean: 21.25058752298355
- stage3_psnr_mean: 21.993137352466583
- condition_residual_l1_mean: 0.33460550885647533
- condition_delta_raw_l1_mean: 0.5193344241008162
- semantic_latent_drop_fraction_mean: 0.0
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailcontrast_m003_w2_ft800_b8.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailcontrast_m003_w2_ft800_b8/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailcontrast_m003_w2_ft800_b8`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_022523-3tekcefr`
