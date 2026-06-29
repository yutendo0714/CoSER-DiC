# 20260629_stage4_detailfilm_stats_hf_dists_ft1000_b8

Date: 2026-06-29T12:08:39

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage4_detailfilm_stats_hf_dists_ft1000_b8 --crop-size 512 --batch-size 8 --num-workers 4 --max-steps 1000 --lr 3e-5 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailfilm_stats_hf_ft1200_b12.pt --init-nonstrict --cod-lite-checkpoint external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.pt --cod-lite-config external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.yaml --cod-lite-repo external/repos/GenCodec/CoD_Lite --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --detail-control-branch --detail-film-modulation --semantic-latent-dropout-prob 0.2 --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 0.7 --condition-cosine-weight 0.25 --condition-channel-stats-weight 0.30 --condition-highfreq-weight 0.08 --condition-highfreq-threshold 0.25 --image-l1-weight 0.12 --lpips-weight 0.06 --dists-weight 0.08 --ms-ssim-weight 0.01 --stage3-l1-guard-weight 0.15 --stage3-mse-guard-weight 0.03 --wandb-mode offline --save-sample-every 250
```

## Summary

- loss_mean: 0.4627743704319
- condition_l1_mean: 0.39842039650678635
- condition_cosine_loss_mean: 0.22042177391052245
- condition_channel_stats_loss_mean: 0.23037420478463172
- condition_highfreq_loss_mean: 0.053713802523910996
- condition_residual_rms_guard_loss_mean: 0.0027719515990584113
- condition_residual_rms_ratio_mean_mean: 0.5034164359867572
- condition_residual_rms_ratio_max_mean: 0.6050006194114685
- detail_contrast_loss_mean: 0.0
- detail_condition_l1_zero_mean: 0.0
- detail_condition_l1_gap_mean: 0.0
- detail_highfreq_residual_loss_mean: 0.0
- detail_highfreq_residual_pred_l1_mean: 0.0
- detail_highfreq_residual_target_l1_mean: 0.0
- base_condition_l1_mean: 0.517362593114376
- condition_l1_delta_vs_base_mean: -0.11894219660758973
- condition_cosine_mean: 0.7795782260894776
- pred_condition_std_mean: 0.7593783926963806
- target_condition_std_mean: 0.81032865691185
- pred_condition_highfreq_mean: 0.22921682135760785
- target_condition_highfreq_mean: 0.24467357224225997
- image_l1_mean: 0.05921892647072673
- lpips_mean: 0.39491096034646034
- dists_loss_mean: 0.2619792237877846
- ms_ssim_loss_mean: 0.2787512845993042
- stage3_l1_guard_mean: 0.005102190826321021
- stage3_mse_guard_mean: 0.001772618786082603
- stage4_psnr_mean: 21.014355266571044
- stage3_psnr_mean: 21.96812466430664
- condition_residual_l1_mean: 0.3458823803961277
- condition_delta_raw_l1_mean: 0.5552053766846656
- semantic_latent_drop_fraction_mean: 0.199625
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailfilm_stats_hf_dists_ft1000_b8.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage4_detailfilm_stats_hf_dists_ft1000_b8/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage4_detailfilm_stats_hf_dists_ft1000_b8`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_115736-lsiqqo98`
