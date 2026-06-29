# 20260629_stage4_detailfilm_dists_perceptual_ft1200_b8

Date: 2026-06-29T12:46:48

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage4_detailfilm_dists_perceptual_ft1200_b8 --crop-size 512 --batch-size 8 --num-workers 4 --max-steps 1200 --lr 2e-5 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailfilm_stats_hf_dists_ft1000_b8.pt --init-nonstrict --cod-lite-checkpoint external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.pt --cod-lite-config external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.yaml --cod-lite-repo external/repos/GenCodec/CoD_Lite --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --detail-control-branch --detail-film-modulation --semantic-latent-dropout-prob 0.2 --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 0.55 --condition-cosine-weight 0.20 --condition-channel-stats-weight 0.25 --condition-highfreq-weight 0.06 --condition-highfreq-threshold 0.25 --image-l1-weight 0.10 --lpips-weight 0.10 --dists-weight 0.14 --ms-ssim-weight 0.02 --stage3-l1-guard-weight 0.22 --stage3-mse-guard-weight 0.05 --wandb-mode offline --save-sample-every 300
```

## Summary

- loss_mean: 0.4083709449817737
- condition_l1_mean: 0.3968163391451041
- condition_cosine_loss_mean: 0.21827524825930594
- condition_channel_stats_loss_mean: 0.2247114897146821
- condition_highfreq_loss_mean: 0.05309511952412625
- condition_residual_rms_guard_loss_mean: 0.0033298086390582664
- condition_residual_rms_ratio_mean_mean: 0.5100811122606198
- condition_residual_rms_ratio_max_mean: 0.6144534682234128
- detail_contrast_loss_mean: 0.0
- detail_condition_l1_zero_mean: 0.0
- detail_condition_l1_gap_mean: 0.0
- detail_highfreq_residual_loss_mean: 0.0
- detail_highfreq_residual_pred_l1_mean: 0.0
- detail_highfreq_residual_target_l1_mean: 0.0
- base_condition_l1_mean: 0.5169338592886925
- condition_l1_delta_vs_base_mean: -0.12011752014358838
- condition_cosine_mean: 0.781724751740694
- pred_condition_std_mean: 0.7615336661040782
- target_condition_std_mean: 0.8103411363065243
- pred_condition_highfreq_mean: 0.22931003344555695
- target_condition_highfreq_mean: 0.24457179407278698
- image_l1_mean: 0.05872842422376076
- lpips_mean: 0.38647968930502735
- dists_loss_mean: 0.25636285964399574
- ms_ssim_loss_mean: 0.27747546101609866
- stage3_l1_guard_mean: 0.004807513870570498
- stage3_mse_guard_mean: 0.0016910090716798247
- stage4_psnr_mean: 21.08017832914988
- stage3_psnr_mean: 21.98875082174937
- condition_residual_l1_mean: 0.35118262633681296
- condition_delta_raw_l1_mean: 0.5710970046867927
- semantic_latent_drop_fraction_mean: 0.1990625
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailfilm_dists_perceptual_ft1200_b8.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage4_detailfilm_dists_perceptual_ft1200_b8/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage4_detailfilm_dists_perceptual_ft1200_b8`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_123334-gdtl1hg7`
