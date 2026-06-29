# 20260629_stage5_d16b3_detailcontrol_fusion_ft400_b4ga2

Date: 2026-06-29T23:46:35

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_2048_seed20260629_d16_b3/reconstructions/manifest.jsonl --run-name 20260629_stage5_d16b3_detailcontrol_fusion_ft400_b4ga2 --crop-size 512 --batch-size 4 --grad-accum-steps 2 --num-workers 4 --max-steps 400 --lr 1.0e-6 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage5_detailcontrol_fusion_lora0156_all_k16_ft600_b4ga2.pt --init-nonstrict --cod-lite-checkpoint external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.pt --cod-lite-config external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.yaml --cod-lite-repo external/repos/GenCodec/CoD_Lite --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --detail-control-branch --detail-control-blocks 2 --detail-control-condition-fusion --detail-film-modulation --semantic-latent-dropout-prob 0.2 --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 0.44 --condition-cosine-weight 0.16 --condition-channel-stats-weight 0.16 --condition-highfreq-weight 0.12 --condition-highfreq-threshold 0.25 --detail-contrast-weight 0.24 --detail-contrast-margin 0.003 --detail-highfreq-residual-weight 0.18 --detail-highfreq-kernel-size 5 --detail-residual-target-weight 0.06 --image-l1-weight 0.14 --lpips-weight 0.18 --dists-weight 0.24 --ms-ssim-weight 0.08 --stage3-l1-guard-weight 0.45 --stage3-mse-guard-weight 0.12 --backbone-lora-plan results/stage5_inspect/20260629_cod_lite_bpp0156_lora_target_plan_rank4.json --backbone-lora-preset denoiser_all --backbone-lora-rank 4 --backbone-lora-alpha 4.0 --backbone-lora-lr 4.0e-6 --backbone-lora-weight-decay 0.0 --train-counted-control-mode condition_residual_affine_basis --control-basis outputs/stage5_control_basis/20260629_lora0156_denoiser_all_postaffine_basis_g32s8_k256_train8192_b12/control_basis.pt --control-grid-size 8 --control-groups 32 --control-basis-components 16 --control-basis-candidate-components 256 --control-basis-selection topk --control-codec huffman --control-huffman-key topk_c256_k16_p99_b4_mulaw16 --control-quantizer mu_law --control-mu 16 --control-bits 4 --control-range 0.695025622844696 --control-affine-groups 16 --control-affine-grid-size 1 --control-affine-gain-range 1.0 --control-affine-bias-range 0.25 --control-scale 1.0 --grad-clip-norm 1.0 --wandb-mode offline --save-sample-every 200
```

## Summary

- loss_mean: 0.4854796739295125
- condition_l1_mean: 0.397691012956202
- adapter_target_condition_l1_mean: 0.39769101291894915
- pre_control_condition_l1_mean: 0.4054998446628451
- control_condition_l1_delta_mean: -0.007808831706643105
- control_payload_bytes_mean: 32.1203125
- control_payload_bpp_512_mean: 0.000980234146118164
- control_grid_abs_mean: 0.33052917633205653
- condition_cosine_loss_mean: 0.21967921532690526
- condition_channel_stats_loss_mean: 0.22063454199582339
- condition_highfreq_loss_mean: 0.05337261626962572
- condition_residual_rms_guard_loss_mean: 0.009980175860049113
- condition_residual_rms_ratio_mean_mean: 0.5216991780698299
- condition_residual_rms_ratio_max_mean: 0.6471627318114043
- detail_contrast_loss_mean: 0.0005399375851266086
- detail_condition_l1_zero_mean: 0.40322268564254043
- detail_condition_l1_gap_mean: 0.005531672723591328
- detail_highfreq_residual_loss_mean: 0.29262563096359373
- detail_highfreq_residual_pred_l1_mean: 0.03376545245060697
- detail_highfreq_residual_target_l1_mean: 0.29288773410022256
- detail_residual_target_loss_mean: 0.5106762694194913
- detail_residual_target_pred_l1_mean: 0.06871204072609544
- detail_residual_target_target_l1_mean: 0.5228298869356514
- base_condition_l1_mean: 0.5185308607295156
- condition_l1_delta_vs_base_mean: -0.12083984777331352
- condition_cosine_mean: 0.7803207846730947
- pred_condition_std_mean: 0.7513451562076807
- target_condition_std_mean: 0.8086724028736353
- pred_condition_highfreq_mean: 0.25648921879008413
- target_condition_highfreq_mean: 0.24553607540205122
- image_l1_mean: 0.05817835059016943
- lpips_mean: 0.34897228222340343
- dists_loss_mean: 0.2293580870144069
- ms_ssim_loss_mean: 0.2703500973433256
- stage3_l1_guard_mean: 0.005247559620474931
- stage3_mse_guard_mean: 0.0016883968727051978
- stage4_psnr_mean: 21.404571290016175
- stage3_psnr_mean: 22.352161071300507
- condition_residual_l1_mean: 0.3537296227179468
- condition_delta_raw_l1_mean: 0.5963301203772425
- semantic_latent_drop_fraction_mean: 0.21125
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage5_d16b3_detailcontrol_fusion_ft400_b4ga2.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage5_d16b3_detailcontrol_fusion_ft400_b4ga2/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage5_d16b3_detailcontrol_fusion_ft400_b4ga2`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_234013-vfta47b0`
