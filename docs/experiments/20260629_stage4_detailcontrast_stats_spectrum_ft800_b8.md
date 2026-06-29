# 20260629_stage4_detailcontrast_stats_spectrum_ft800_b8

Date: 2026-06-29T15:47:13

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage4_detailcontrast_stats_spectrum_ft800_b8 --crop-size 512 --batch-size 8 --num-workers 4 --max-steps 800 --lr 1.0e-5 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8.pt --init-nonstrict --cod-lite-checkpoint external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.pt --cod-lite-config external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.yaml --cod-lite-repo external/repos/GenCodec/CoD_Lite --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --detail-control-branch --detail-film-modulation --semantic-latent-dropout-prob 0.2 --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 0.45 --condition-cosine-weight 0.20 --condition-channel-stats-weight 0.55 --condition-highfreq-weight 0.18 --condition-highfreq-threshold 0.25 --detail-contrast-weight 0.12 --detail-contrast-margin 0.002 --detail-highfreq-residual-weight 0.08 --detail-highfreq-kernel-size 5 --image-l1-weight 0.12 --lpips-weight 0.08 --dists-weight 0.12 --ms-ssim-weight 0.03 --stage3-l1-guard-weight 0.32 --stage3-mse-guard-weight 0.08 --wandb-mode offline --save-sample-every 200
```

## Summary

- loss_mean: 0.4507163942977786
- condition_l1_mean: 0.3997861198708415
- condition_cosine_loss_mean: 0.21853617899119854
- condition_channel_stats_loss_mean: 0.2118800433538854
- condition_highfreq_loss_mean: 0.051357550048269335
- condition_residual_rms_guard_loss_mean: 0.004030491649953092
- condition_residual_rms_ratio_mean_mean: 0.5207199789211154
- condition_residual_rms_ratio_max_mean: 0.6244539841264486
- detail_contrast_loss_mean: 0.0018014576390851288
- detail_condition_l1_zero_mean: 0.3999973238259554
- detail_condition_l1_gap_mean: 0.00021120395511388778
- detail_highfreq_residual_loss_mean: 0.2910123509913683
- detail_highfreq_residual_pred_l1_mean: 0.01039509329944849
- detail_highfreq_residual_target_l1_mean: 0.29079438477754593
- base_condition_l1_mean: 0.5174244556576013
- condition_l1_delta_vs_base_mean: -0.11763833578675985
- condition_cosine_mean: 0.7814638210088014
- pred_condition_std_mean: 0.7745237211883068
- target_condition_std_mean: 0.809734931141138
- pred_condition_highfreq_mean: 0.2355397069454193
- target_condition_highfreq_mean: 0.24501644235104322
- image_l1_mean: 0.05871983530465513
- lpips_mean: 0.38380819201469424
- dists_loss_mean: 0.2517205960303545
- ms_ssim_loss_mean: 0.2759559740126133
- stage3_l1_guard_mean: 0.004560964395932387
- stage3_mse_guard_mean: 0.0016771024555782789
- stage4_psnr_mean: 21.07004717350006
- stage3_psnr_mean: 21.964708380699157
- condition_residual_l1_mean: 0.35990718301385644
- condition_delta_raw_l1_mean: 0.594766705557704
- semantic_latent_drop_fraction_mean: 0.20140625
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailcontrast_stats_spectrum_ft800_b8.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage4_detailcontrast_stats_spectrum_ft800_b8/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage4_detailcontrast_stats_spectrum_ft800_b8`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_153750-fvibbwte`
