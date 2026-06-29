# 20260629_stage5_controlaware_adapter_lora0156_all_k16_ft400_b4ga2

Date: 2026-06-29T21:09:56

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage5_controlaware_adapter_lora0156_all_k16_ft400_b4ga2 --crop-size 512 --batch-size 4 --grad-accum-steps 2 --num-workers 4 --max-steps 400 --lr 1.0e-6 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage5_detailtarget_lora0156_denoiser_all_r4_balanced_ft900_b4ga2.pt --init-nonstrict --cod-lite-checkpoint external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.pt --cod-lite-config external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.yaml --cod-lite-repo external/repos/GenCodec/CoD_Lite --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --detail-control-branch --detail-film-modulation --semantic-latent-dropout-prob 0.2 --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 0.44 --condition-cosine-weight 0.16 --condition-channel-stats-weight 0.16 --condition-highfreq-weight 0.12 --condition-highfreq-threshold 0.25 --detail-contrast-weight 0.24 --detail-contrast-margin 0.003 --detail-highfreq-residual-weight 0.18 --detail-highfreq-kernel-size 5 --detail-residual-target-weight 0.06 --image-l1-weight 0.14 --lpips-weight 0.18 --dists-weight 0.24 --ms-ssim-weight 0.08 --stage3-l1-guard-weight 0.45 --stage3-mse-guard-weight 0.12 --backbone-lora-plan results/stage5_inspect/20260629_cod_lite_bpp0156_lora_target_plan_rank4.json --backbone-lora-preset denoiser_all --backbone-lora-rank 4 --backbone-lora-alpha 4.0 --backbone-lora-lr 4.0e-6 --backbone-lora-weight-decay 0.0 --train-counted-control-mode condition_residual_affine_basis --control-basis outputs/stage5_control_basis/20260629_lora0156_denoiser_all_postaffine_basis_g32s8_k256_train8192_b12/control_basis.pt --control-grid-size 8 --control-groups 32 --control-basis-components 16 --control-basis-candidate-components 256 --control-basis-selection topk --control-codec huffman --control-huffman-key topk_c256_k16_p99_b4_mulaw16 --control-quantizer mu_law --control-mu 16 --control-bits 4 --control-range 0.695025622844696 --control-affine-groups 16 --control-affine-grid-size 1 --control-affine-gain-range 1.0 --control-affine-bias-range 0.25 --control-scale 1.0 --grad-clip-norm 1.0 --wandb-mode offline --save-sample-every 200
```

## Summary

- loss_mean: 0.4745954290032387
- condition_l1_mean: 0.387305931635201
- pre_control_condition_l1_mean: 0.39341330736875535
- control_condition_l1_delta_mean: -0.006107375733554363
- control_payload_bytes_mean: 32.4359375
- control_payload_bpp_512_mean: 0.0009898662567138672
- control_grid_abs_mean: 0.3117500964738429
- condition_cosine_loss_mean: 0.20639449566602708
- condition_channel_stats_loss_mean: 0.2054950744472444
- condition_highfreq_loss_mean: 0.050522492132149634
- condition_residual_rms_guard_loss_mean: 0.011174142543575382
- condition_residual_rms_ratio_mean_mean: 0.5341312634572387
- condition_residual_rms_ratio_max_mean: 0.6578546611219644
- detail_contrast_loss_mean: 0.0
- detail_condition_l1_zero_mean: 0.40037091106176376
- detail_condition_l1_gap_mean: 0.013064979426562786
- detail_highfreq_residual_loss_mean: 0.2883025878481567
- detail_highfreq_residual_pred_l1_mean: 0.03367222278844565
- detail_highfreq_residual_target_l1_mean: 0.29057426895946264
- detail_residual_target_loss_mean: 0.493478694036603
- detail_residual_target_pred_l1_mean: 0.10187552941963077
- detail_residual_target_target_l1_mean: 0.5179257822036744
- base_condition_l1_mean: 0.5179286738857627
- condition_l1_delta_vs_base_mean: -0.13062274225056172
- condition_cosine_mean: 0.7936055043339729
- pred_condition_std_mean: 0.7649807366728782
- target_condition_std_mean: 0.8103754936158657
- pred_condition_highfreq_mean: 0.23174822764471173
- target_condition_highfreq_mean: 0.24437829036265613
- image_l1_mean: 0.058794306146446616
- lpips_mean: 0.34299827152863144
- dists_loss_mean: 0.23312617810443043
- ms_ssim_loss_mean: 0.2805898827314377
- stage3_l1_guard_mean: 0.004772671247774269
- stage3_mse_guard_mean: 0.0016377942502731457
- stage4_psnr_mean: 21.148469891548157
- stage3_psnr_mean: 21.9891392993927
- condition_residual_l1_mean: 0.36411800464615224
- condition_delta_raw_l1_mean: 0.5532064478099347
- semantic_latent_drop_fraction_mean: 0.21125
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage5_controlaware_adapter_lora0156_all_k16_ft400_b4ga2.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage5_controlaware_adapter_lora0156_all_k16_ft400_b4ga2/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage5_controlaware_adapter_lora0156_all_k16_ft400_b4ga2`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_210337-wlj0jb4y`
