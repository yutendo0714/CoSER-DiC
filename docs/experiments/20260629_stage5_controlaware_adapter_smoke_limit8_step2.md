# 20260629_stage5_controlaware_adapter_smoke_limit8_step2

Date: 2026-06-29T21:03:03

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage5_controlaware_adapter_smoke_limit8_step2 --crop-size 512 --limit 8 --batch-size 1 --grad-accum-steps 1 --num-workers 0 --max-steps 2 --lr 1.0e-6 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage5_detailtarget_lora0156_denoiser_all_r4_balanced_ft900_b4ga2.pt --init-nonstrict --cod-lite-checkpoint external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.pt --cod-lite-config external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.yaml --cod-lite-repo external/repos/GenCodec/CoD_Lite --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --detail-control-branch --detail-film-modulation --semantic-latent-dropout-prob 0.2 --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 0.44 --condition-cosine-weight 0.16 --condition-channel-stats-weight 0.16 --condition-highfreq-weight 0.12 --condition-highfreq-threshold 0.25 --detail-contrast-weight 0.24 --detail-contrast-margin 0.003 --detail-highfreq-residual-weight 0.18 --detail-highfreq-kernel-size 5 --detail-residual-target-weight 0.06 --image-l1-weight 0.14 --lpips-weight 0.18 --dists-weight 0.24 --ms-ssim-weight 0.08 --stage3-l1-guard-weight 0.45 --stage3-mse-guard-weight 0.12 --backbone-lora-plan results/stage5_inspect/20260629_cod_lite_bpp0156_lora_target_plan_rank4.json --backbone-lora-preset denoiser_all --backbone-lora-rank 4 --backbone-lora-alpha 4.0 --backbone-lora-lr 4.0e-6 --backbone-lora-weight-decay 0.0 --train-counted-control-mode condition_residual_affine_basis --control-basis outputs/stage5_control_basis/20260629_lora0156_denoiser_all_postaffine_basis_g32s8_k256_train8192_b12/control_basis.pt --control-grid-size 8 --control-groups 32 --control-basis-components 16 --control-basis-candidate-components 256 --control-basis-selection topk --control-codec huffman --control-huffman-key topk_c256_k16_p99_b4_mulaw16 --control-quantizer mu_law --control-mu 16 --control-bits 4 --control-range 0.695025622844696 --control-affine-groups 16 --control-affine-grid-size 1 --control-affine-gain-range 1.0 --control-affine-bias-range 0.25 --control-scale 1.0 --grad-clip-norm 1.0 --wandb-mode offline --save-sample-every 0
```

## Summary

- loss_mean: 0.4713120460510254
- condition_l1_mean: 0.3828458935022354
- pre_control_condition_l1_mean: 0.3870130628347397
- control_condition_l1_delta_mean: -0.0041671693325042725
- control_payload_bytes_mean: 32.0
- control_payload_bpp_512_mean: 0.0009765625
- control_grid_abs_mean: 0.288084477186203
- condition_cosine_loss_mean: 0.21872487664222717
- condition_channel_stats_loss_mean: 0.20552537590265274
- condition_highfreq_loss_mean: 0.04803331382572651
- condition_residual_rms_guard_loss_mean: 0.008365850895643234
- condition_residual_rms_ratio_mean_mean: 0.5075876712799072
- condition_residual_rms_ratio_max_mean: 0.5075876712799072
- detail_contrast_loss_mean: 0.0
- detail_condition_l1_zero_mean: 0.39435696601867676
- detail_condition_l1_gap_mean: 0.011511072516441345
- detail_highfreq_residual_loss_mean: 0.2831474095582962
- detail_highfreq_residual_pred_l1_mean: 0.027541182935237885
- detail_highfreq_residual_target_l1_mean: 0.28519582748413086
- detail_residual_target_loss_mean: 0.4798208475112915
- detail_residual_target_pred_l1_mean: 0.08248919993638992
- detail_residual_target_target_l1_mean: 0.5047193169593811
- base_condition_l1_mean: 0.5047231316566467
- condition_l1_delta_vs_base_mean: -0.12187723815441132
- condition_cosine_mean: 0.7812751233577728
- pred_condition_std_mean: 0.7628141045570374
- target_condition_std_mean: 0.81228107213974
- pred_condition_highfreq_mean: 0.24309220165014267
- target_condition_highfreq_mean: 0.25155578553676605
- image_l1_mean: 0.06693767756223679
- lpips_mean: 0.3294256404042244
- dists_loss_mean: 0.22081336379051208
- ms_ssim_loss_mean: 0.3092324435710907
- stage3_l1_guard_mean: 0.0061285048723220825
- stage3_mse_guard_mean: 0.002504180185496807
- stage4_psnr_mean: 18.859058380126953
- stage3_psnr_mean: 19.782304763793945
- condition_residual_l1_mean: 0.33934633433818817
- condition_delta_raw_l1_mean: 0.4942793548107147
- semantic_latent_drop_fraction_mean: 0.5
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage5_controlaware_adapter_smoke_limit8_step2.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage5_controlaware_adapter_smoke_limit8_step2/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage5_controlaware_adapter_smoke_limit8_step2`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_210259-b4k4nx37`
