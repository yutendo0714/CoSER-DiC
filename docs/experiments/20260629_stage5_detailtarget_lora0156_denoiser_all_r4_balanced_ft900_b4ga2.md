# 20260629_stage5_detailtarget_lora0156_denoiser_all_r4_balanced_ft900_b4ga2

Date: 2026-06-29T19:42:21

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage5_detailtarget_lora0156_denoiser_all_r4_balanced_ft900_b4ga2 --crop-size 512 --batch-size 4 --grad-accum-steps 2 --num-workers 4 --max-steps 900 --lr 1.5e-6 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailtarget_perceptual_long_ft1800_b8.pt --init-nonstrict --cod-lite-checkpoint external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.pt --cod-lite-config external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.yaml --cod-lite-repo external/repos/GenCodec/CoD_Lite --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --detail-control-branch --detail-film-modulation --semantic-latent-dropout-prob 0.2 --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 0.44 --condition-cosine-weight 0.16 --condition-channel-stats-weight 0.16 --condition-highfreq-weight 0.12 --condition-highfreq-threshold 0.25 --detail-contrast-weight 0.24 --detail-contrast-margin 0.003 --detail-highfreq-residual-weight 0.18 --detail-highfreq-kernel-size 5 --detail-residual-target-weight 0.06 --image-l1-weight 0.14 --lpips-weight 0.18 --dists-weight 0.24 --ms-ssim-weight 0.08 --stage3-l1-guard-weight 0.45 --stage3-mse-guard-weight 0.12 --backbone-lora-plan results/stage5_inspect/20260629_cod_lite_bpp0156_lora_target_plan_rank4.json --backbone-lora-preset denoiser_all --backbone-lora-rank 4 --backbone-lora-alpha 4.0 --backbone-lora-lr 8.0e-6 --backbone-lora-weight-decay 0.0 --grad-clip-norm 1.0 --wandb-mode offline --save-sample-every 450
```

## Summary

- loss_mean: 0.48852970254090095
- condition_l1_mean: 0.39393728327420025
- condition_cosine_loss_mean: 0.21564340376191668
- condition_channel_stats_loss_mean: 0.2214060135351287
- condition_highfreq_loss_mean: 0.051312389938781656
- condition_residual_rms_guard_loss_mean: 0.003625236362781254
- condition_residual_rms_ratio_mean_mean: 0.5124653530783123
- condition_residual_rms_ratio_max_mean: 0.5892508890397019
- detail_contrast_loss_mean: 0.0005942922613273064
- detail_condition_l1_zero_mean: 0.3995074293017387
- detail_condition_l1_gap_mean: 0.005570146027538511
- detail_highfreq_residual_loss_mean: 0.29017419656117754
- detail_highfreq_residual_pred_l1_mean: 0.023178509669895802
- detail_highfreq_residual_target_l1_mean: 0.2906609654592143
- detail_residual_target_loss_mean: 0.5074317405124505
- detail_residual_target_pred_l1_mean: 0.054484776806914145
- detail_residual_target_target_l1_mean: 0.5176583141419623
- base_condition_l1_mean: 0.5176612190074391
- condition_l1_delta_vs_base_mean: -0.12372393573323885
- condition_cosine_mean: 0.7843565962380833
- pred_condition_std_mean: 0.7601783628927337
- target_condition_std_mean: 0.8105569903055827
- pred_condition_highfreq_mean: 0.2334848728444841
- target_condition_highfreq_mean: 0.2445897138118744
- image_l1_mean: 0.058472176532571515
- lpips_mean: 0.3635606281293763
- dists_loss_mean: 0.24244276776081986
- ms_ssim_loss_mean: 0.27874913665983414
- stage3_l1_guard_mean: 0.004427146184801434
- stage3_mse_guard_mean: 0.0015548919182037935
- stage4_psnr_mean: 21.160771373112997
- stage3_psnr_mean: 21.982966657214696
- condition_residual_l1_mean: 0.353376522709926
- condition_delta_raw_l1_mean: 0.5785268482234743
- semantic_latent_drop_fraction_mean: 0.19694444444444445
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage5_detailtarget_lora0156_denoiser_all_r4_balanced_ft900_b4ga2.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage5_detailtarget_lora0156_denoiser_all_r4_balanced_ft900_b4ga2/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage5_detailtarget_lora0156_denoiser_all_r4_balanced_ft900_b4ga2`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_192840-k6yhyf4i`
