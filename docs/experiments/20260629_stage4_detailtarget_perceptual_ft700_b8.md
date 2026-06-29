# 20260629_stage4_detailtarget_perceptual_ft700_b8

Date: 2026-06-29T16:26:34

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage4_detailtarget_perceptual_ft700_b8 --crop-size 512 --batch-size 8 --num-workers 4 --max-steps 700 --lr 8.0e-6 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8.pt --init-nonstrict --cod-lite-checkpoint external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.pt --cod-lite-config external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.yaml --cod-lite-repo external/repos/GenCodec/CoD_Lite --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --detail-control-branch --detail-film-modulation --semantic-latent-dropout-prob 0.2 --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 0.45 --condition-cosine-weight 0.15 --condition-channel-stats-weight 0.15 --condition-highfreq-weight 0.10 --condition-highfreq-threshold 0.25 --detail-contrast-weight 0.22 --detail-contrast-margin 0.003 --detail-highfreq-residual-weight 0.16 --detail-highfreq-kernel-size 5 --detail-residual-target-weight 0.03 --image-l1-weight 0.10 --lpips-weight 0.18 --dists-weight 0.24 --ms-ssim-weight 0.02 --stage3-l1-guard-weight 0.28 --stage3-mse-guard-weight 0.06 --wandb-mode offline --save-sample-every 350
```

## Summary

- loss_mean: 0.45219496084111077
- condition_l1_mean: 0.39445937382323404
- condition_cosine_loss_mean: 0.21680223609719956
- condition_channel_stats_loss_mean: 0.22366399473377638
- condition_highfreq_loss_mean: 0.052124841575111665
- condition_residual_rms_guard_loss_mean: 0.0037881386977954077
- condition_residual_rms_ratio_mean_mean: 0.5168312825901168
- condition_residual_rms_ratio_max_mean: 0.6217624714119093
- detail_contrast_loss_mean: 0.0025988410892231126
- detail_condition_l1_zero_mean: 0.3948643034696579
- detail_condition_l1_gap_mean: 0.0004049296464238848
- detail_highfreq_residual_loss_mean: 0.290922304051263
- detail_highfreq_residual_pred_l1_mean: 0.010396553086383003
- detail_highfreq_residual_target_l1_mean: 0.2907986746089799
- detail_residual_target_loss_mean: 0.5148879700473377
- detail_residual_target_pred_l1_mean: 0.022875931702022043
- detail_residual_target_target_l1_mean: 0.5176074263453484
- base_condition_l1_mean: 0.5176103386708668
- condition_l1_delta_vs_base_mean: -0.12315096484763281
- condition_cosine_mean: 0.7831977639028004
- pred_condition_std_mean: 0.7565783057894025
- target_condition_std_mean: 0.8098462841340474
- pred_condition_highfreq_mean: 0.2315880514894213
- target_condition_highfreq_mean: 0.24477939507790974
- image_l1_mean: 0.05863427491060325
- lpips_mean: 0.37780341386795047
- dists_loss_mean: 0.2502199540393693
- ms_ssim_loss_mean: 0.27832870270524707
- stage3_l1_guard_mean: 0.004490369141900113
- stage3_mse_guard_mean: 0.0015860883925675547
- stage4_psnr_mean: 21.12730563027518
- stage3_psnr_mean: 21.971743281228203
- condition_residual_l1_mean: 0.3565965586049216
- condition_delta_raw_l1_mean: 0.5866351624046053
- semantic_latent_drop_fraction_mean: 0.2017857142857143
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailtarget_perceptual_ft700_b8.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage4_detailtarget_perceptual_ft700_b8/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage4_detailtarget_perceptual_ft700_b8`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_161821-jcuwu6j2`
