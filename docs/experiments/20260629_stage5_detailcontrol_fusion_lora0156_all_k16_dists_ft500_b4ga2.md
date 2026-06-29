# 20260629_stage5_detailcontrol_fusion_lora0156_all_k16_dists_ft500_b4ga2

Date: 2026-06-29T22:48:50

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage5_detailcontrol_fusion_lora0156_all_k16_dists_ft500_b4ga2 --crop-size 512 --batch-size 4 --grad-accum-steps 2 --num-workers 4 --max-steps 500 --lr 1.0e-6 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage5_detailtarget_lora0156_denoiser_all_r4_balanced_ft900_b4ga2.pt --init-nonstrict --cod-lite-checkpoint external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.pt --cod-lite-config external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.yaml --cod-lite-repo external/repos/GenCodec/CoD_Lite --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --detail-control-branch --detail-control-blocks 2 --detail-control-condition-fusion --detail-film-modulation --semantic-latent-dropout-prob 0.2 --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 0.42 --pre-control-condition-l1-weight 0.05 --condition-cosine-weight 0.16 --condition-channel-stats-weight 0.16 --condition-highfreq-weight 0.14 --condition-highfreq-threshold 0.25 --detail-contrast-weight 0.28 --detail-contrast-margin 0.003 --detail-highfreq-residual-weight 0.24 --detail-highfreq-kernel-size 5 --detail-residual-target-weight 0.10 --image-l1-weight 0.16 --lpips-weight 0.12 --dists-weight 0.36 --ms-ssim-weight 0.10 --stage3-l1-guard-weight 0.50 --stage3-mse-guard-weight 0.14 --backbone-lora-plan results/stage5_inspect/20260629_cod_lite_bpp0156_lora_target_plan_rank4.json --backbone-lora-preset denoiser_all --backbone-lora-rank 4 --backbone-lora-alpha 4.0 --backbone-lora-lr 4.0e-6 --backbone-lora-weight-decay 0.0 --train-counted-control-mode condition_residual_affine_basis --control-basis outputs/stage5_control_basis/20260629_lora0156_denoiser_all_postaffine_basis_g32s8_k256_train8192_b12/control_basis.pt --control-grid-size 8 --control-groups 32 --control-basis-components 16 --control-basis-candidate-components 256 --control-basis-selection topk --control-codec huffman --control-huffman-key topk_c256_k16_p99_b4_mulaw16 --control-quantizer mu_law --control-mu 16 --control-bits 4 --control-range 0.695025622844696 --control-affine-groups 16 --control-affine-grid-size 1 --control-affine-gain-range 1.0 --control-affine-bias-range 0.25 --control-scale 1.0 --grad-clip-norm 1.0 --wandb-mode offline --save-sample-every 250
```

## Summary

- loss_mean: 0.5383924639225006
- condition_l1_mean: 0.38697160020470617
- adapter_target_condition_l1_mean: 0.3869716002643108
- pre_control_condition_l1_mean: 0.3930634950697422
- control_condition_l1_delta_mean: -0.00609189486503601
- control_payload_bytes_mean: 32.39725
- control_payload_bpp_512_mean: 0.0009886856079101562
- control_grid_abs_mean: 0.3122544872313738
- condition_cosine_loss_mean: 0.20559916418790816
- condition_channel_stats_loss_mean: 0.20500982573628426
- condition_highfreq_loss_mean: 0.05045690386742353
- condition_residual_rms_guard_loss_mean: 0.010530096253704215
- condition_residual_rms_ratio_mean_mean: 0.5304592845141888
- condition_residual_rms_ratio_max_mean: 0.6514547073841095
- detail_contrast_loss_mean: 0.00027408023551106454
- detail_condition_l1_zero_mean: 0.39337356749176977
- detail_condition_l1_gap_mean: 0.006401967227458954
- detail_highfreq_residual_loss_mean: 0.28795783445239065
- detail_highfreq_residual_pred_l1_mean: 0.024674245544709265
- detail_highfreq_residual_target_l1_mean: 0.28866130696237086
- detail_residual_target_loss_mean: 0.490295568138361
- detail_residual_target_pred_l1_mean: 0.057798065451905134
- detail_residual_target_target_l1_mean: 0.5010974051058292
- base_condition_l1_mean: 0.516320595651865
- condition_l1_delta_vs_base_mean: -0.1293489954471588
- condition_cosine_mean: 0.7944008358120919
- pred_condition_std_mean: 0.7656560552120208
- target_condition_std_mean: 0.8110694338083267
- pred_condition_highfreq_mean: 0.2321653372645378
- target_condition_highfreq_mean: 0.24431075143814088
- image_l1_mean: 0.05835733674839139
- lpips_mean: 0.34317076541483404
- dists_loss_mean: 0.2350847601145506
- ms_ssim_loss_mean: 0.2768240554332733
- stage3_l1_guard_mean: 0.004379026930779219
- stage3_mse_guard_mean: 0.0015252776411653032
- stage4_psnr_mean: 21.174971017837525
- stage3_psnr_mean: 21.95722961807251
- condition_residual_l1_mean: 0.36122901007533076
- condition_delta_raw_l1_mean: 0.5388341305851936
- semantic_latent_drop_fraction_mean: 0.208
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage5_detailcontrol_fusion_lora0156_all_k16_dists_ft500_b4ga2.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage5_detailcontrol_fusion_lora0156_all_k16_dists_ft500_b4ga2/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage5_detailcontrol_fusion_lora0156_all_k16_dists_ft500_b4ga2`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_224055-5kzdkj4c`
