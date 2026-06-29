# 20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8

Date: 2026-06-29T13:10:51

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8 --crop-size 512 --batch-size 8 --num-workers 4 --max-steps 800 --lr 1.5e-5 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailfilm_dists_perceptual_ft1200_b8.pt --init-nonstrict --cod-lite-checkpoint external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.pt --cod-lite-config external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.yaml --cod-lite-repo external/repos/GenCodec/CoD_Lite --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --detail-control-branch --detail-film-modulation --semantic-latent-dropout-prob 0.2 --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 0.50 --condition-cosine-weight 0.18 --condition-channel-stats-weight 0.22 --condition-highfreq-weight 0.06 --condition-highfreq-threshold 0.25 --detail-contrast-weight 0.08 --detail-contrast-margin 0.002 --detail-highfreq-residual-weight 0.04 --detail-highfreq-kernel-size 5 --image-l1-weight 0.10 --lpips-weight 0.10 --dists-weight 0.14 --ms-ssim-weight 0.02 --stage3-l1-guard-weight 0.24 --stage3-mse-guard-weight 0.05 --wandb-mode offline --save-sample-every 200
```

## Summary

- loss_mean: 0.3879762331023812
- condition_l1_mean: 0.3961131491884589
- condition_cosine_loss_mean: 0.21774819262325765
- condition_channel_stats_loss_mean: 0.22298470562323927
- condition_highfreq_loss_mean: 0.052849892396479844
- condition_residual_rms_guard_loss_mean: 0.0033886836376700556
- condition_residual_rms_ratio_mean_mean: 0.5126051985844969
- condition_residual_rms_ratio_max_mean: 0.6155677803978324
- detail_contrast_loss_mean: 0.0018659464770462363
- detail_condition_l1_zero_mean: 0.39625574063509705
- detail_condition_l1_gap_mean: 0.0001425914466381073
- detail_highfreq_residual_loss_mean: 0.2910236497595906
- detail_highfreq_residual_pred_l1_mean: 0.0097322234447347
- detail_highfreq_residual_target_l1_mean: 0.29079438477754593
- base_condition_l1_mean: 0.5174244556576013
- condition_l1_delta_vs_base_mean: -0.12131130646914244
- condition_cosine_mean: 0.7822518073767424
- pred_condition_std_mean: 0.760923152640462
- target_condition_std_mean: 0.809734931141138
- pred_condition_highfreq_mean: 0.22978141080588102
- target_condition_highfreq_mean: 0.24501644235104322
- image_l1_mean: 0.058711330271326
- lpips_mean: 0.384380707629025
- dists_loss_mean: 0.2547943723574281
- ms_ssim_loss_mean: 0.27770437762141226
- stage3_l1_guard_mean: 0.0045475721903494555
- stage3_mse_guard_mean: 0.001624205545449513
- stage4_psnr_mean: 21.098280320167543
- stage3_psnr_mean: 21.964708380699157
- condition_residual_l1_mean: 0.3532107526063919
- condition_delta_raw_l1_mean: 0.5761362943798304
- semantic_latent_drop_fraction_mean: 0.20140625
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_130130-y1bgqifw`
