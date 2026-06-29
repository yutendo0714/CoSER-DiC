# 20260629_stage4_detailcontrast_statslite_ft500_b8

Date: 2026-06-29T15:55:51

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage4_detailcontrast_statslite_ft500_b8 --crop-size 512 --batch-size 8 --num-workers 4 --max-steps 500 --lr 8.0e-6 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8.pt --init-nonstrict --cod-lite-checkpoint external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.pt --cod-lite-config external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.yaml --cod-lite-repo external/repos/GenCodec/CoD_Lite --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --detail-control-branch --detail-film-modulation --semantic-latent-dropout-prob 0.2 --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 0.65 --condition-cosine-weight 0.20 --condition-channel-stats-weight 0.34 --condition-highfreq-weight 0.10 --condition-highfreq-threshold 0.25 --detail-contrast-weight 0.10 --detail-contrast-margin 0.002 --detail-highfreq-residual-weight 0.06 --detail-highfreq-kernel-size 5 --image-l1-weight 0.12 --lpips-weight 0.10 --dists-weight 0.14 --ms-ssim-weight 0.03 --stage3-l1-guard-weight 0.34 --stage3-mse-guard-weight 0.08 --wandb-mode offline --save-sample-every 250
```

## Summary

- loss_mean: 0.48860613864660263
- condition_l1_mean: 0.3956661288738251
- condition_cosine_loss_mean: 0.21709409272670746
- condition_channel_stats_loss_mean: 0.21816426995396615
- condition_highfreq_loss_mean: 0.05216746505349874
- condition_residual_rms_guard_loss_mean: 0.0035465530564069924
- condition_residual_rms_ratio_mean_mean: 0.5147934206128121
- condition_residual_rms_ratio_max_mean: 0.6180678956508636
- detail_contrast_loss_mean: 0.0017613600962795317
- detail_condition_l1_zero_mean: 0.3959156261086464
- detail_condition_l1_gap_mean: 0.00024949723482131956
- detail_highfreq_residual_loss_mean: 0.29090125000476835
- detail_highfreq_residual_pred_l1_mean: 0.010083583804778754
- detail_highfreq_residual_target_l1_mean: 0.2906986095905304
- base_condition_l1_mean: 0.5177554019093513
- condition_l1_delta_vs_base_mean: -0.12208927303552627
- condition_cosine_mean: 0.7829059072732926
- pred_condition_std_mean: 0.7629861488342286
- target_condition_std_mean: 0.809076189994812
- pred_condition_highfreq_mean: 0.23096614092588424
- target_condition_highfreq_mean: 0.24467843639850617
- image_l1_mean: 0.058542310893535614
- lpips_mean: 0.3842264829277992
- dists_loss_mean: 0.2543338105678558
- ms_ssim_loss_mean: 0.27688032245635985
- stage3_l1_guard_mean: 0.004390464516822248
- stage3_mse_guard_mean: 0.001598869479028508
- stage4_psnr_mean: 21.115413314819335
- stage3_psnr_mean: 21.961987438201906
- condition_residual_l1_mean: 0.35491925436258315
- condition_delta_raw_l1_mean: 0.5812180260419846
- semantic_latent_drop_fraction_mean: 0.2025
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailcontrast_statslite_ft500_b8.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage4_detailcontrast_statslite_ft500_b8/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage4_detailcontrast_statslite_ft500_b8`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_154957-prhfih90`
