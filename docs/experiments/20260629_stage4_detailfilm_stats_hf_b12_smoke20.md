# 20260629_stage4_detailfilm_stats_hf_b12_smoke20

Date: 2026-06-29T11:09:50

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage4_detailfilm_stats_hf_b12_smoke20 --crop-size 512 --limit 512 --batch-size 12 --num-workers 4 --max-steps 20 --lr 5e-5 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailfilm_ft800_b8.pt --init-nonstrict --cod-lite-checkpoint external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.pt --cod-lite-config external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.yaml --cod-lite-repo external/repos/GenCodec/CoD_Lite --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --detail-control-branch --detail-film-modulation --semantic-latent-dropout-prob 0.2 --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 1.0 --condition-cosine-weight 0.3 --condition-channel-stats-weight 0.35 --condition-highfreq-weight 0.08 --condition-highfreq-threshold 0.25 --image-l1-weight 0.20 --lpips-weight 0.04 --dists-weight 0.0 --ms-ssim-weight 0.02 --stage3-l1-guard-weight 0.25 --stage3-mse-guard-weight 0.05 --stage3-guard-margin 0.0 --wandb-mode offline --save-sample-every 0
```

## Summary

- loss_mean: 0.603338435292244
- condition_l1_mean: 0.40254706889390945
- condition_cosine_loss_mean: 0.23442898988723754
- condition_channel_stats_loss_mean: 0.25568425953388213
- condition_highfreq_loss_mean: 0.056700137071311475
- condition_residual_rms_guard_loss_mean: 0.003154348974931054
- condition_residual_rms_ratio_mean_mean: 0.5037662535905838
- condition_residual_rms_ratio_max_mean: 0.6314342349767685
- detail_contrast_loss_mean: 0.0
- detail_condition_l1_zero_mean: 0.0
- detail_condition_l1_gap_mean: 0.0
- detail_highfreq_residual_loss_mean: 0.0
- detail_highfreq_residual_pred_l1_mean: 0.0
- detail_highfreq_residual_target_l1_mean: 0.0
- base_condition_l1_mean: 0.5206171110272407
- condition_l1_delta_vs_base_mean: -0.1180700421333313
- condition_cosine_mean: 0.7655710101127624
- pred_condition_std_mean: 0.7336778491735458
- target_condition_std_mean: 0.802196791768074
- pred_condition_highfreq_mean: 0.2300726018846035
- target_condition_highfreq_mean: 0.24739149287343026
- image_l1_mean: 0.06153220292180776
- lpips_mean: 0.41718015968799593
- dists_loss_mean: 0.0
- ms_ssim_loss_mean: 0.29130900502204893
- stage3_l1_guard_mean: 0.006094624730758369
- stage3_mse_guard_mean: 0.0018738819693680854
- stage4_psnr_mean: 20.837203693389892
- stage3_psnr_mean: 21.865968036651612
- condition_residual_l1_mean: 0.3411312192678452
- condition_delta_raw_l1_mean: 0.5395888358354568
- semantic_latent_drop_fraction_mean: 0.25416664183139803
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailfilm_stats_hf_b12_smoke20.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage4_detailfilm_stats_hf_b12_smoke20/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage4_detailfilm_stats_hf_b12_smoke20`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_110931-ygf2f8jt`
