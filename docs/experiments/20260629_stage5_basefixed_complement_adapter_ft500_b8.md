# 20260629_stage5_basefixed_complement_adapter_ft500_b8

Date: 2026-06-29T21:48:40

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage5_basefixed_complement_adapter_ft500_b8 --crop-size 512 --batch-size 8 --num-workers 4 --max-steps 500 --lr 2.0e-6 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailtarget_perceptual_long_ft1800_b8.pt --init-nonstrict --cod-lite-checkpoint external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.pt --cod-lite-config external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.yaml --cod-lite-repo external/repos/GenCodec/CoD_Lite --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --detail-control-branch --detail-film-modulation --semantic-latent-dropout-prob 0.2 --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 0.44 --condition-cosine-weight 0.16 --condition-channel-stats-weight 0.16 --condition-highfreq-weight 0.12 --condition-highfreq-threshold 0.25 --detail-contrast-weight 0.24 --detail-contrast-margin 0.003 --detail-highfreq-residual-weight 0.18 --detail-highfreq-kernel-size 5 --detail-residual-target-weight 0.06 --image-l1-weight 0.12 --lpips-weight 0.20 --dists-weight 0.26 --ms-ssim-weight 0.05 --stage3-l1-guard-weight 0.35 --stage3-mse-guard-weight 0.09 --train-counted-control-mode condition_base_affine_basis --control-basis outputs/stage5_control_basis/20260629_detailtarget_perceptual_long_basefixed_basis_g32s8_k256_train8192_b12/control_basis.pt --control-grid-size 8 --control-groups 32 --control-basis-components 32 --control-basis-candidate-components 256 --control-basis-selection topk --control-codec huffman --control-huffman-key topk_c256_k32_p99_b4_mulaw16 --control-quantizer mu_law --control-mu 16 --control-bits 4 --control-range 0.9494848847389221 --control-affine-groups 16 --control-affine-grid-size 1 --control-affine-gain-range 1.0 --control-affine-bias-range 0.25 --control-scale 1.0 --grad-clip-norm 1.0 --wandb-mode offline --save-sample-every 250
```

## Summary

- loss_mean: 0.4915079241394997
- condition_l1_mean: 0.3948496644496918
- adapter_target_condition_l1_mean: 0.39484966480731964
- pre_control_condition_l1_mean: 0.39519255912303924
- control_condition_l1_delta_mean: -0.00034289467334747316
- control_payload_bytes_mean: 48.06075
- control_payload_bpp_512_mean: 0.0014666976928710937
- control_grid_abs_mean: 0.2938596602976322
- condition_cosine_loss_mean: 0.21480965852737427
- condition_channel_stats_loss_mean: 0.2258084643483162
- condition_highfreq_loss_mean: 0.051116961367428305
- condition_residual_rms_guard_loss_mean: 0.008113484633970075
- condition_residual_rms_ratio_mean_mean: 0.5504199981093406
- condition_residual_rms_ratio_max_mean: 0.6697772114276886
- detail_contrast_loss_mean: 0.0004146360857412219
- detail_condition_l1_zero_mean: 0.3995318596363068
- detail_condition_l1_gap_mean: 0.004682194828987122
- detail_highfreq_residual_loss_mean: 0.2900829614400864
- detail_highfreq_residual_pred_l1_mean: 0.022913550823926924
- detail_highfreq_residual_target_l1_mean: 0.2906763055920601
- detail_residual_target_loss_mean: 0.49854390913248064
- detail_residual_target_pred_l1_mean: 0.05306819133460522
- detail_residual_target_target_l1_mean: 0.5077910896539688
- base_condition_l1_mean: 0.5177554019093513
- condition_l1_delta_vs_base_mean: -0.12290573745965958
- condition_cosine_mean: 0.7851903414726258
- pred_condition_std_mean: 0.7637951650619507
- target_condition_std_mean: 0.809076189994812
- pred_condition_highfreq_mean: 0.23105723693966865
- target_condition_highfreq_mean: 0.24467843639850617
- image_l1_mean: 0.05855887886881828
- lpips_mean: 0.36560536235570906
- dists_loss_mean: 0.24299750965833664
- ms_ssim_loss_mean: 0.2779912315607071
- stage3_l1_guard_mean: 0.004409779158886522
- stage3_mse_guard_mean: 0.0015953662941465155
- stage4_psnr_mean: 21.136815422058106
- stage3_psnr_mean: 21.961987438201906
- condition_residual_l1_mean: 0.37447108852863314
- condition_delta_raw_l1_mean: 0.5634028822779655
- semantic_latent_drop_fraction_mean: 0.2025
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage5_basefixed_complement_adapter_ft500_b8.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage5_basefixed_complement_adapter_ft500_b8/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage5_basefixed_complement_adapter_ft500_b8`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_214235-4knwrxce`
