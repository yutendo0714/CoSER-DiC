# 20260629_stage4_detailfilm_stats_hf_ft1200_b12

Date: 2026-06-29T11:26:02

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage4_detailfilm_stats_hf_ft1200_b12 --crop-size 512 --batch-size 12 --num-workers 4 --max-steps 1200 --lr 5e-5 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailfilm_ft800_b8.pt --init-nonstrict --cod-lite-checkpoint external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.pt --cod-lite-config external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.yaml --cod-lite-repo external/repos/GenCodec/CoD_Lite --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --detail-control-branch --detail-film-modulation --semantic-latent-dropout-prob 0.2 --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 1.0 --condition-cosine-weight 0.3 --condition-channel-stats-weight 0.35 --condition-highfreq-weight 0.08 --condition-highfreq-threshold 0.25 --image-l1-weight 0.20 --lpips-weight 0.04 --dists-weight 0.0 --ms-ssim-weight 0.02 --stage3-l1-guard-weight 0.25 --stage3-mse-guard-weight 0.05 --stage3-guard-margin 0.0 --wandb-mode offline --save-sample-every 300
```

## Summary

- loss_mean: 0.5899492910007635
- condition_l1_mean: 0.39975087945659954
- condition_cosine_loss_mean: 0.2232889187335968
- condition_channel_stats_loss_mean: 0.23978204359610875
- condition_highfreq_loss_mean: 0.05489074285763006
- condition_residual_rms_guard_loss_mean: 0.002220716022746198
- condition_residual_rms_ratio_mean_mean: 0.49578784639636675
- condition_residual_rms_ratio_max_mean: 0.6106975585222244
- detail_contrast_loss_mean: 0.0
- detail_condition_l1_zero_mean: 0.0
- detail_condition_l1_gap_mean: 0.0
- detail_highfreq_residual_loss_mean: 0.0
- detail_highfreq_residual_pred_l1_mean: 0.0
- detail_highfreq_residual_target_l1_mean: 0.0
- base_condition_l1_mean: 0.5173621097952128
- condition_l1_delta_vs_base_mean: -0.11761123033861319
- condition_cosine_mean: 0.7767110812664032
- pred_condition_std_mean: 0.7525122234721978
- target_condition_std_mean: 0.8103155382474263
- pred_condition_highfreq_mean: 0.22785188374420007
- target_condition_highfreq_mean: 0.24485860149065652
- image_l1_mean: 0.05926630630468329
- lpips_mean: 0.4017900657157103
- dists_loss_mean: 0.0
- ms_ssim_loss_mean: 0.2790367211898168
- stage3_l1_guard_mean: 0.005215065929417809
- stage3_mse_guard_mean: 0.0017479277984724225
- stage4_psnr_mean: 21.020731072425843
- stage3_psnr_mean: 21.97126987616221
- condition_residual_l1_mean: 0.33990436516702177
- condition_delta_raw_l1_mean: 0.5377472912768523
- semantic_latent_drop_fraction_mean: 0.2053472023208936
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailfilm_stats_hf_ft1200_b12.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage4_detailfilm_stats_hf_ft1200_b12/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage4_detailfilm_stats_hf_ft1200_b12`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_111014-ekqu327t`
