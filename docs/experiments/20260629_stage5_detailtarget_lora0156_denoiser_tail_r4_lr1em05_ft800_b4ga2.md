# 20260629_stage5_detailtarget_lora0156_denoiser_tail_r4_lr1em05_ft800_b4ga2

Date: 2026-06-29T19:02:26

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage5_detailtarget_lora0156_denoiser_tail_r4_lr1em05_ft800_b4ga2 --crop-size 512 --batch-size 4 --grad-accum-steps 2 --num-workers 4 --max-steps 800 --lr 2.0e-6 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailtarget_perceptual_long_ft1800_b8.pt --init-nonstrict --cod-lite-checkpoint external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.pt --cod-lite-config external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.yaml --cod-lite-repo external/repos/GenCodec/CoD_Lite --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --detail-control-branch --detail-film-modulation --semantic-latent-dropout-prob 0.2 --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 0.44 --condition-cosine-weight 0.16 --condition-channel-stats-weight 0.16 --condition-highfreq-weight 0.12 --condition-highfreq-threshold 0.25 --detail-contrast-weight 0.24 --detail-contrast-margin 0.003 --detail-highfreq-residual-weight 0.18 --detail-highfreq-kernel-size 5 --detail-residual-target-weight 0.06 --image-l1-weight 0.10 --lpips-weight 0.20 --dists-weight 0.26 --ms-ssim-weight 0.03 --stage3-l1-guard-weight 0.30 --stage3-mse-guard-weight 0.07 --backbone-lora-plan results/stage5_inspect/20260629_cod_lite_bpp0156_lora_target_plan_rank4.json --backbone-lora-preset denoiser_tail --backbone-lora-rank 4 --backbone-lora-alpha 4.0 --backbone-lora-lr 1.0e-5 --backbone-lora-weight-decay 0.0 --grad-clip-norm 1.0 --wandb-mode offline --save-sample-every 400
```

## Summary

- loss_mean: 0.4843109990097582
- condition_l1_mean: 0.3939919807575643
- condition_cosine_loss_mean: 0.21575061030685902
- condition_channel_stats_loss_mean: 0.22135881253518164
- condition_highfreq_loss_mean: 0.05149389436934143
- condition_residual_rms_guard_loss_mean: 0.0037932730511160907
- condition_residual_rms_ratio_mean_mean: 0.5156011379323899
- condition_residual_rms_ratio_max_mean: 0.592027612477541
- detail_contrast_loss_mean: 0.000664009320025798
- detail_condition_l1_zero_mean: 0.39941071001812817
- detail_condition_l1_gap_mean: 0.005418729260563851
- detail_highfreq_residual_loss_mean: 0.29010498507879673
- detail_highfreq_residual_pred_l1_mean: 0.023322587014408782
- detail_highfreq_residual_target_l1_mean: 0.29058202866464855
- detail_residual_target_loss_mean: 0.507123408317566
- detail_residual_target_pred_l1_mean: 0.054731893639545885
- detail_residual_target_target_l1_mean: 0.5172310929745436
- base_condition_l1_mean: 0.5172339883819222
- condition_l1_delta_vs_base_mean: -0.12324200762435794
- condition_cosine_mean: 0.784249389693141
- pred_condition_std_mean: 0.7598774663358927
- target_condition_std_mean: 0.8104487691447139
- pred_condition_highfreq_mean: 0.2325976146105677
- target_condition_highfreq_mean: 0.24462408388964832
- image_l1_mean: 0.05851880983565934
- lpips_mean: 0.36546699804253874
- dists_loss_mean: 0.2432032334152609
- ms_ssim_loss_mean: 0.27934526406228544
- stage3_l1_guard_mean: 0.004534376356750727
- stage3_mse_guard_mean: 0.0016185379902708518
- stage4_psnr_mean: 21.12340633392334
- stage3_psnr_mean: 21.977552123069763
- condition_residual_l1_mean: 0.3559447166696191
- condition_delta_raw_l1_mean: 0.5856881909631193
- semantic_latent_drop_fraction_mean: 0.1984375
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage5_detailtarget_lora0156_denoiser_tail_r4_lr1em05_ft800_b4ga2.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage5_detailtarget_lora0156_denoiser_tail_r4_lr1em05_ft800_b4ga2/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage5_detailtarget_lora0156_denoiser_tail_r4_lr1em05_ft800_b4ga2`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_185217-ae4ld0we`
