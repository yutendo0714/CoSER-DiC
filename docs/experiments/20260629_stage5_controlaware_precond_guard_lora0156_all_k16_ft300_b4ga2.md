# 20260629_stage5_controlaware_precond_guard_lora0156_all_k16_ft300_b4ga2

Date: 2026-06-29T21:27:59

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage5_controlaware_precond_guard_lora0156_all_k16_ft300_b4ga2 --crop-size 512 --batch-size 4 --grad-accum-steps 2 --num-workers 4 --max-steps 300 --lr 8.0e-7 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage5_detailtarget_lora0156_denoiser_all_r4_balanced_ft900_b4ga2.pt --init-nonstrict --cod-lite-checkpoint external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.pt --cod-lite-config external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.yaml --cod-lite-repo external/repos/GenCodec/CoD_Lite --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --detail-control-branch --detail-film-modulation --semantic-latent-dropout-prob 0.2 --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 0.40 --pre-control-condition-l1-weight 0.18 --condition-cosine-weight 0.16 --condition-channel-stats-weight 0.16 --condition-highfreq-weight 0.12 --condition-highfreq-threshold 0.25 --detail-contrast-weight 0.24 --detail-contrast-margin 0.003 --detail-highfreq-residual-weight 0.18 --detail-highfreq-kernel-size 5 --detail-residual-target-weight 0.06 --image-l1-weight 0.14 --lpips-weight 0.18 --dists-weight 0.24 --ms-ssim-weight 0.08 --stage3-l1-guard-weight 0.45 --stage3-mse-guard-weight 0.12 --backbone-lora-plan results/stage5_inspect/20260629_cod_lite_bpp0156_lora_target_plan_rank4.json --backbone-lora-preset denoiser_all --backbone-lora-rank 4 --backbone-lora-alpha 4.0 --backbone-lora-lr 3.0e-6 --backbone-lora-weight-decay 0.0 --train-counted-control-mode condition_residual_affine_basis --control-basis outputs/stage5_control_basis/20260629_lora0156_denoiser_all_postaffine_basis_g32s8_k256_train8192_b12/control_basis.pt --control-grid-size 8 --control-groups 32 --control-basis-components 16 --control-basis-candidate-components 256 --control-basis-selection topk --control-codec huffman --control-huffman-key topk_c256_k16_p99_b4_mulaw16 --control-quantizer mu_law --control-mu 16 --control-bits 4 --control-range 0.695025622844696 --control-affine-groups 16 --control-affine-grid-size 1 --control-affine-gain-range 1.0 --control-affine-bias-range 0.25 --control-scale 1.0 --grad-clip-norm 1.0 --wandb-mode offline --save-sample-every 150
```

## Summary

- loss_mean: 0.5297474092245102
- condition_l1_mean: 0.38689020603895186
- pre_control_condition_l1_mean: 0.392875205129385
- control_condition_l1_delta_mean: -0.005984999090433121
- control_payload_bytes_mean: 32.475833333333334
- control_payload_bpp_512_mean: 0.0009910837809244792
- control_grid_abs_mean: 0.31011899722119174
- condition_cosine_loss_mean: 0.20625189165274302
- condition_channel_stats_loss_mean: 0.2059535545607408
- condition_highfreq_loss_mean: 0.05054436442752679
- condition_residual_rms_guard_loss_mean: 0.011166933948668963
- condition_residual_rms_ratio_mean_mean: 0.5346975526710351
- condition_residual_rms_ratio_max_mean: 0.6569501352310181
- detail_contrast_loss_mean: 0.0
- detail_condition_l1_zero_mean: 0.39983889569838843
- detail_condition_l1_gap_mean: 0.012948689659436545
- detail_highfreq_residual_loss_mean: 0.28826972377796967
- detail_highfreq_residual_pred_l1_mean: 0.03351947710849345
- detail_highfreq_residual_target_l1_mean: 0.2904931107660135
- detail_residual_target_loss_mean: 0.49377479260166485
- detail_residual_target_pred_l1_mean: 0.10142520545671384
- detail_residual_target_target_l1_mean: 0.5177867988745372
- base_condition_l1_mean: 0.5177896904945374
- condition_l1_delta_vs_base_mean: -0.13089948445558547
- condition_cosine_mean: 0.793748108347257
- pred_condition_std_mean: 0.763513535062472
- target_condition_std_mean: 0.810441760122776
- pred_condition_highfreq_mean: 0.23107471284766992
- target_condition_highfreq_mean: 0.24386580395201843
- image_l1_mean: 0.05876809353008866
- lpips_mean: 0.3428471060842276
- dists_loss_mean: 0.23314279516537983
- ms_ssim_loss_mean: 0.2810503793756167
- stage3_l1_guard_mean: 0.004811672123614699
- stage3_mse_guard_mean: 0.001641535331242873
- stage4_psnr_mean: 21.16909515698751
- stage3_psnr_mean: 22.013718032836913
- condition_residual_l1_mean: 0.3647079132994016
- condition_delta_raw_l1_mean: 0.5563447462022304
- semantic_latent_drop_fraction_mean: 0.21916666666666668
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage5_controlaware_precond_guard_lora0156_all_k16_ft300_b4ga2.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage5_controlaware_precond_guard_lora0156_all_k16_ft300_b4ga2/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage5_controlaware_precond_guard_lora0156_all_k16_ft300_b4ga2`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_212316-6ipxs6u6`
