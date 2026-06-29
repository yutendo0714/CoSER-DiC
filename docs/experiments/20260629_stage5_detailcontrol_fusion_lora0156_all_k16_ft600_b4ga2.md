# 20260629_stage5_detailcontrol_fusion_lora0156_all_k16_ft600_b4ga2

Date: 2026-06-29T22:37:31

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage5_detailcontrol_fusion_lora0156_all_k16_ft600_b4ga2 --crop-size 512 --batch-size 4 --grad-accum-steps 2 --num-workers 4 --max-steps 600 --lr 1.0e-6 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage5_detailtarget_lora0156_denoiser_all_r4_balanced_ft900_b4ga2.pt --init-nonstrict --cod-lite-checkpoint external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.pt --cod-lite-config external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.yaml --cod-lite-repo external/repos/GenCodec/CoD_Lite --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --detail-control-branch --detail-control-blocks 2 --detail-control-condition-fusion --detail-film-modulation --semantic-latent-dropout-prob 0.2 --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 0.44 --condition-cosine-weight 0.16 --condition-channel-stats-weight 0.16 --condition-highfreq-weight 0.12 --condition-highfreq-threshold 0.25 --detail-contrast-weight 0.24 --detail-contrast-margin 0.003 --detail-highfreq-residual-weight 0.18 --detail-highfreq-kernel-size 5 --detail-residual-target-weight 0.06 --image-l1-weight 0.14 --lpips-weight 0.18 --dists-weight 0.24 --ms-ssim-weight 0.08 --stage3-l1-guard-weight 0.45 --stage3-mse-guard-weight 0.12 --backbone-lora-plan results/stage5_inspect/20260629_cod_lite_bpp0156_lora_target_plan_rank4.json --backbone-lora-preset denoiser_all --backbone-lora-rank 4 --backbone-lora-alpha 4.0 --backbone-lora-lr 4.0e-6 --backbone-lora-weight-decay 0.0 --train-counted-control-mode condition_residual_affine_basis --control-basis outputs/stage5_control_basis/20260629_lora0156_denoiser_all_postaffine_basis_g32s8_k256_train8192_b12/control_basis.pt --control-grid-size 8 --control-groups 32 --control-basis-components 16 --control-basis-candidate-components 256 --control-basis-selection topk --control-codec huffman --control-huffman-key topk_c256_k16_p99_b4_mulaw16 --control-quantizer mu_law --control-mu 16 --control-bits 4 --control-range 0.695025622844696 --control-affine-groups 16 --control-affine-grid-size 1 --control-affine-gain-range 1.0 --control-affine-bias-range 0.25 --control-scale 1.0 --grad-clip-norm 1.0 --wandb-mode offline --save-sample-every 300
```

## Summary

- loss_mean: 0.47349766197303933
- condition_l1_mean: 0.3868547892818848
- adapter_target_condition_l1_mean: 0.3868547886610031
- pre_control_condition_l1_mean: 0.3929830077290535
- control_condition_l1_delta_mean: -0.006128218447168668
- control_payload_bytes_mean: 32.39208333333333
- control_payload_bpp_512_mean: 0.0009885279337565104
- control_grid_abs_mean: 0.3121243682131171
- condition_cosine_loss_mean: 0.205593285014232
- condition_channel_stats_loss_mean: 0.2049545823658506
- condition_highfreq_loss_mean: 0.05059181721570591
- condition_residual_rms_guard_loss_mean: 0.010865447371002888
- condition_residual_rms_ratio_mean_mean: 0.531600992033879
- condition_residual_rms_ratio_max_mean: 0.6535794199009737
- detail_contrast_loss_mean: 0.00031324934020328024
- detail_condition_l1_zero_mean: 0.39288974481324357
- detail_condition_l1_gap_mean: 0.006034956152240436
- detail_highfreq_residual_loss_mean: 0.2880738733833035
- detail_highfreq_residual_pred_l1_mean: 0.024328476134687662
- detail_highfreq_residual_target_l1_mean: 0.28873810374488434
- detail_residual_target_loss_mean: 0.49167821124196054
- detail_residual_target_pred_l1_mean: 0.05697733376175165
- detail_residual_target_target_l1_mean: 0.5022100254893302
- base_condition_l1_mean: 0.5166312913844983
- condition_l1_delta_vs_base_mean: -0.12977650210261346
- condition_cosine_mean: 0.794406714985768
- pred_condition_std_mean: 0.7658315238853296
- target_condition_std_mean: 0.8111011568705241
- pred_condition_highfreq_mean: 0.23178954157978296
- target_condition_highfreq_mean: 0.24441644368072352
- image_l1_mean: 0.058588487130279344
- lpips_mean: 0.34124090443054833
- dists_loss_mean: 0.23282471215973297
- ms_ssim_loss_mean: 0.27886551852027575
- stage3_l1_guard_mean: 0.00464263422996737
- stage3_mse_guard_mean: 0.001596672662490164
- stage4_psnr_mean: 21.15649006843567
- stage3_psnr_mean: 21.976746966044107
- condition_residual_l1_mean: 0.3621695716927449
- condition_delta_raw_l1_mean: 0.5446955814460913
- semantic_latent_drop_fraction_mean: 0.205
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage5_detailcontrol_fusion_lora0156_all_k16_ft600_b4ga2.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage5_detailcontrol_fusion_lora0156_all_k16_ft600_b4ga2/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage5_detailcontrol_fusion_lora0156_all_k16_ft600_b4ga2`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_222800-k9nuks3k`
