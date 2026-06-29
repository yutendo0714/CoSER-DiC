# 20260629_stage5_detailtarget_lora0078_denoiser_all_r4_balanced_ft700_b4ga2

Date: 2026-06-29T20:15:04

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage5_detailtarget_lora0078_denoiser_all_r4_balanced_ft700_b4ga2 --crop-size 512 --batch-size 4 --grad-accum-steps 2 --num-workers 4 --max-steps 700 --lr 1.5e-6 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailtarget_perceptual_long_ft1800_b8.pt --init-nonstrict --cod-lite-checkpoint external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0078.pt --cod-lite-config external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0078.yaml --cod-lite-repo external/repos/GenCodec/CoD_Lite --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --detail-control-branch --detail-film-modulation --semantic-latent-dropout-prob 0.2 --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 0.44 --condition-cosine-weight 0.16 --condition-channel-stats-weight 0.16 --condition-highfreq-weight 0.12 --condition-highfreq-threshold 0.25 --detail-contrast-weight 0.24 --detail-contrast-margin 0.003 --detail-highfreq-residual-weight 0.18 --detail-highfreq-kernel-size 5 --detail-residual-target-weight 0.06 --image-l1-weight 0.14 --lpips-weight 0.18 --dists-weight 0.24 --ms-ssim-weight 0.08 --stage3-l1-guard-weight 0.45 --stage3-mse-guard-weight 0.12 --backbone-lora-plan results/stage5_inspect/20260629_cod_lite_bpp0156_lora_target_plan_rank4.json --backbone-lora-preset denoiser_all --backbone-lora-rank 4 --backbone-lora-alpha 4.0 --backbone-lora-lr 8.0e-6 --backbone-lora-weight-decay 0.0 --grad-clip-norm 1.0 --wandb-mode offline --save-sample-every 350
```

## Summary

- loss_mean: 0.540589285258736
- condition_l1_mean: 0.46019789308309555
- condition_cosine_loss_mean: 0.24016049121107375
- condition_channel_stats_loss_mean: 0.2966234772971698
- condition_highfreq_loss_mean: 0.03762913707510701
- condition_residual_rms_guard_loss_mean: 0.00016838592297719034
- condition_residual_rms_ratio_mean_mean: 0.4192168997653893
- condition_residual_rms_ratio_max_mean: 0.4794445919139045
- detail_contrast_loss_mean: 0.0016040338266507855
- detail_condition_l1_zero_mean: 0.46231990505542075
- detail_condition_l1_gap_mean: 0.0021220119723251887
- detail_highfreq_residual_loss_mean: 0.2617950230304684
- detail_highfreq_residual_pred_l1_mean: 0.01958550890175892
- detail_highfreq_residual_target_l1_mean: 0.2621481260338
- detail_residual_target_loss_mean: 0.5159646878923688
- detail_residual_target_pred_l1_mean: 0.04647877757038389
- detail_residual_target_target_l1_mean: 0.5211783948966435
- base_condition_l1_mean: 0.5211783948966435
- condition_l1_delta_vs_base_mean: -0.06098050181354795
- condition_cosine_mean: 0.7598395087889263
- pred_condition_std_mean: 0.8523355386938367
- target_condition_std_mean: 0.8745915201732091
- pred_condition_highfreq_mean: 0.1659712748761688
- target_condition_highfreq_mean: 0.15413406801543064
- image_l1_mean: 0.06291674801813704
- lpips_mean: 0.3891910398219313
- dists_loss_mean: 0.2559027454576322
- ms_ssim_loss_mean: 0.30835209978478295
- stage3_l1_guard_mean: 0.008809769563376903
- stage3_mse_guard_mean: 0.0026947749231476335
- stage4_psnr_mean: 20.609870832988193
- stage3_psnr_mean: 21.991802166530064
- condition_residual_l1_mean: 0.30324801151241576
- condition_delta_raw_l1_mean: 0.45100200261388507
- semantic_latent_drop_fraction_mean: 0.2007142857142857
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage5_detailtarget_lora0078_denoiser_all_r4_balanced_ft700_b4ga2.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage5_detailtarget_lora0078_denoiser_all_r4_balanced_ft700_b4ga2/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage5_detailtarget_lora0078_denoiser_all_r4_balanced_ft700_b4ga2`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_200434-yh8gh58b`
