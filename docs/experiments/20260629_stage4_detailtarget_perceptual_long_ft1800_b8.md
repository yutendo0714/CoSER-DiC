# 20260629_stage4_detailtarget_perceptual_long_ft1800_b8

Date: 2026-06-29T17:03:43

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage4_detailtarget_perceptual_long_ft1800_b8 --crop-size 512 --batch-size 8 --num-workers 4 --max-steps 1800 --lr 4.0e-6 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailtarget_perceptual_ft700_b8.pt --init-nonstrict --cod-lite-checkpoint external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.pt --cod-lite-config external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.yaml --cod-lite-repo external/repos/GenCodec/CoD_Lite --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --detail-control-branch --detail-film-modulation --semantic-latent-dropout-prob 0.2 --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 0.44 --condition-cosine-weight 0.16 --condition-channel-stats-weight 0.16 --condition-highfreq-weight 0.12 --condition-highfreq-threshold 0.25 --detail-contrast-weight 0.24 --detail-contrast-margin 0.003 --detail-highfreq-residual-weight 0.18 --detail-highfreq-kernel-size 5 --detail-residual-target-weight 0.06 --image-l1-weight 0.10 --lpips-weight 0.20 --dists-weight 0.26 --ms-ssim-weight 0.03 --stage3-l1-guard-weight 0.30 --stage3-mse-guard-weight 0.07 --wandb-mode offline --save-sample-every 900
```

## Summary

- loss_mean: 0.48778788967265024
- condition_l1_mean: 0.3942041675746441
- condition_cosine_loss_mean: 0.21602509750260246
- condition_channel_stats_loss_mean: 0.2222423323078288
- condition_highfreq_loss_mean: 0.05160390923420588
- condition_residual_rms_guard_loss_mean: 0.004038737511982823
- condition_residual_rms_ratio_mean_mean: 0.5182760464648406
- condition_residual_rms_ratio_max_mean: 0.6263234969973565
- detail_contrast_loss_mean: 0.0014709037912285163
- detail_condition_l1_zero_mean: 0.3963238351378176
- detail_condition_l1_gap_mean: 0.002119667563173506
- detail_highfreq_residual_loss_mean: 0.29040754036770927
- detail_highfreq_residual_pred_l1_mean: 0.016127789304074314
- detail_highfreq_residual_target_l1_mean: 0.2904627384079827
- detail_residual_target_loss_mean: 0.5113348406222131
- detail_residual_target_pred_l1_mean: 0.03705366122743322
- detail_residual_target_target_l1_mean: 0.5171290087037617
- base_condition_l1_mean: 0.5171318911347125
- condition_l1_delta_vs_base_mean: -0.12292772356006834
- condition_cosine_mean: 0.7839749024973975
- pred_condition_std_mean: 0.7588297264443503
- target_condition_std_mean: 0.8105532130599022
- pred_condition_highfreq_mean: 0.23281449261638854
- target_condition_highfreq_mean: 0.24462421423859065
- image_l1_mean: 0.058164419082717765
- lpips_mean: 0.3736568286518256
- dists_loss_mean: 0.24790255304839875
- ms_ssim_loss_mean: 0.27674321479267544
- stage3_l1_guard_mean: 0.0043417699676420955
- stage3_mse_guard_mean: 0.0015576551382158262
- stage4_psnr_mean: 21.176675475438437
- stage3_psnr_mean: 22.00832611295912
- condition_residual_l1_mean: 0.35802889751063455
- condition_delta_raw_l1_mean: 0.5914282280537817
- semantic_latent_drop_fraction_mean: 0.19770833333333335
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailtarget_perceptual_long_ft1800_b8.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage4_detailtarget_perceptual_long_ft1800_b8/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage4_detailtarget_perceptual_long_ft1800_b8`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_164245-gc4fz0k8`
