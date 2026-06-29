# 20260629_stage4_detailtarget_ft700_b8

Date: 2026-06-29T16:11:50

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage4_detailtarget_ft700_b8 --crop-size 512 --batch-size 8 --num-workers 4 --max-steps 700 --lr 8.0e-6 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8.pt --init-nonstrict --cod-lite-checkpoint external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.pt --cod-lite-config external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.yaml --cod-lite-repo external/repos/GenCodec/CoD_Lite --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --detail-control-branch --detail-film-modulation --semantic-latent-dropout-prob 0.2 --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 0.58 --condition-cosine-weight 0.20 --condition-channel-stats-weight 0.20 --condition-highfreq-weight 0.08 --condition-highfreq-threshold 0.25 --detail-contrast-weight 0.18 --detail-contrast-margin 0.003 --detail-highfreq-residual-weight 0.10 --detail-highfreq-kernel-size 5 --detail-residual-target-weight 0.06 --image-l1-weight 0.12 --lpips-weight 0.12 --dists-weight 0.16 --ms-ssim-weight 0.03 --stage3-l1-guard-weight 0.34 --stage3-mse-guard-weight 0.08 --wandb-mode offline --save-sample-every 350
```

## Summary

- loss_mean: 0.4844467727627073
- condition_l1_mean: 0.3938143005541393
- condition_cosine_loss_mean: 0.21602518788405828
- condition_channel_stats_loss_mean: 0.22308570908648628
- condition_highfreq_loss_mean: 0.05248279619429793
- condition_residual_rms_guard_loss_mean: 0.0033580223956245652
- condition_residual_rms_ratio_mean_mean: 0.5117685814840452
- condition_residual_rms_ratio_max_mean: 0.6154986393025943
- detail_contrast_loss_mean: 0.0025272952246346643
- detail_condition_l1_zero_mean: 0.39429381996393204
- detail_condition_l1_gap_mean: 0.0004795194097927638
- detail_highfreq_residual_loss_mean: 0.2909795416678701
- detail_highfreq_residual_pred_l1_mean: 0.010466194857311036
- detail_highfreq_residual_target_l1_mean: 0.2907986746089799
- detail_residual_target_loss_mean: 0.5145276154790606
- detail_residual_target_pred_l1_mean: 0.023350287305989436
- detail_residual_target_target_l1_mean: 0.5176074263453484
- base_condition_l1_mean: 0.5176103386708668
- condition_l1_delta_vs_base_mean: -0.12379603811672756
- condition_cosine_mean: 0.7839748121159418
- pred_condition_std_mean: 0.7575346755981446
- target_condition_std_mean: 0.8098462841340474
- pred_condition_highfreq_mean: 0.22990983117903982
- target_condition_highfreq_mean: 0.24477939507790974
- image_l1_mean: 0.05841182015836239
- lpips_mean: 0.38254248844725747
- dists_loss_mean: 0.25488089399678365
- ms_ssim_loss_mean: 0.2772663391487939
- stage3_l1_guard_mean: 0.004274279816142683
- stage3_mse_guard_mean: 0.0015307110994555323
- stage4_psnr_mean: 21.155257110595702
- stage3_psnr_mean: 21.971743281228203
- condition_residual_l1_mean: 0.35259561253445487
- condition_delta_raw_l1_mean: 0.5748518828409058
- semantic_latent_drop_fraction_mean: 0.2017857142857143
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailtarget_ft700_b8.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage4_detailtarget_ft700_b8/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage4_detailtarget_ft700_b8`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_160339-a756rsaq`
