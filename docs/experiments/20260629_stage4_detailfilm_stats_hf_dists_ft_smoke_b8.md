# 20260629_stage4_detailfilm_stats_hf_dists_ft_smoke_b8

Date: 2026-06-29T11:57:14

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage4_detailfilm_stats_hf_dists_ft_smoke_b8 --crop-size 512 --batch-size 8 --num-workers 4 --max-steps 10 --lr 3e-5 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailfilm_stats_hf_ft1200_b12.pt --init-nonstrict --cod-lite-checkpoint external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.pt --cod-lite-config external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.yaml --cod-lite-repo external/repos/GenCodec/CoD_Lite --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --detail-control-branch --detail-film-modulation --semantic-latent-dropout-prob 0.2 --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 0.7 --condition-cosine-weight 0.25 --condition-channel-stats-weight 0.30 --condition-highfreq-weight 0.08 --condition-highfreq-threshold 0.25 --image-l1-weight 0.12 --lpips-weight 0.06 --dists-weight 0.08 --ms-ssim-weight 0.01 --stage3-l1-guard-weight 0.15 --stage3-mse-guard-weight 0.03 --wandb-mode offline --save-sample-every 0
```

## Summary

- loss_mean: 0.46931045055389403
- condition_l1_mean: 0.402068567276001
- condition_cosine_loss_mean: 0.230960214138031
- condition_channel_stats_loss_mean: 0.23022280037403106
- condition_highfreq_loss_mean: 0.05486766472458839
- condition_residual_rms_guard_loss_mean: 0.0028612265014089644
- condition_residual_rms_ratio_mean_mean: 0.5181394159793854
- condition_residual_rms_ratio_max_mean: 0.6119000494480134
- detail_contrast_loss_mean: 0.0
- detail_condition_l1_zero_mean: 0.0
- detail_condition_l1_gap_mean: 0.0
- detail_highfreq_residual_loss_mean: 0.0
- detail_highfreq_residual_pred_l1_mean: 0.0
- detail_highfreq_residual_target_l1_mean: 0.0
- base_condition_l1_mean: 0.5229111671447754
- condition_l1_delta_vs_base_mean: -0.12084259986877441
- condition_cosine_mean: 0.769039785861969
- pred_condition_std_mean: 0.7412923276424408
- target_condition_std_mean: 0.7965653002262115
- pred_condition_highfreq_mean: 0.23764808177948
- target_condition_highfreq_mean: 0.2503862053155899
- image_l1_mean: 0.06077391728758812
- lpips_mean: 0.40845479667186735
- dists_loss_mean: 0.2641755372285843
- ms_ssim_loss_mean: 0.287290495634079
- stage3_l1_guard_mean: 0.005362733220681548
- stage3_mse_guard_mean: 0.0018211163114756346
- stage4_psnr_mean: 20.850205993652345
- stage3_psnr_mean: 21.810724639892577
- condition_residual_l1_mean: 0.35114761888980867
- condition_delta_raw_l1_mean: 0.5658774852752686
- semantic_latent_drop_fraction_mean: 0.225
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailfilm_stats_hf_dists_ft_smoke_b8.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage4_detailfilm_stats_hf_dists_ft_smoke_b8/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage4_detailfilm_stats_hf_dists_ft_smoke_b8`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_115704-c5jrz2iw`
