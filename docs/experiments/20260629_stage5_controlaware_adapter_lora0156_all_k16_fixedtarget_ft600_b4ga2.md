# 20260629_stage5_controlaware_adapter_lora0156_all_k16_fixedtarget_ft600_b4ga2

Date: 2026-06-29T22:04:56

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage5_controlaware_adapter_lora0156_all_k16_fixedtarget_ft600_b4ga2 --crop-size 512 --batch-size 4 --grad-accum-steps 2 --num-workers 4 --max-steps 600 --lr 1.0e-6 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage5_detailtarget_lora0156_denoiser_all_r4_balanced_ft900_b4ga2.pt --init-nonstrict --cod-lite-checkpoint external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.pt --cod-lite-config external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.yaml --cod-lite-repo external/repos/GenCodec/CoD_Lite --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --detail-control-branch --detail-film-modulation --semantic-latent-dropout-prob 0.2 --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 0.44 --condition-cosine-weight 0.16 --condition-channel-stats-weight 0.16 --condition-highfreq-weight 0.12 --condition-highfreq-threshold 0.25 --detail-contrast-weight 0.24 --detail-contrast-margin 0.003 --detail-highfreq-residual-weight 0.18 --detail-highfreq-kernel-size 5 --detail-residual-target-weight 0.06 --image-l1-weight 0.14 --lpips-weight 0.18 --dists-weight 0.24 --ms-ssim-weight 0.08 --stage3-l1-guard-weight 0.45 --stage3-mse-guard-weight 0.12 --backbone-lora-plan results/stage5_inspect/20260629_cod_lite_bpp0156_lora_target_plan_rank4.json --backbone-lora-preset denoiser_all --backbone-lora-rank 4 --backbone-lora-alpha 4.0 --backbone-lora-lr 4.0e-6 --backbone-lora-weight-decay 0.0 --train-counted-control-mode condition_residual_affine_basis --control-basis outputs/stage5_control_basis/20260629_lora0156_denoiser_all_postaffine_basis_g32s8_k256_train8192_b12/control_basis.pt --control-grid-size 8 --control-groups 32 --control-basis-components 16 --control-basis-candidate-components 256 --control-basis-selection topk --control-codec huffman --control-huffman-key topk_c256_k16_p99_b4_mulaw16 --control-quantizer mu_law --control-mu 16 --control-bits 4 --control-range 0.695025622844696 --control-affine-groups 16 --control-affine-grid-size 1 --control-affine-gain-range 1.0 --control-affine-bias-range 0.25 --control-scale 1.0 --grad-clip-norm 1.0 --wandb-mode offline --save-sample-every 300
```

## Summary

- loss_mean: 0.4739732664823532
- condition_l1_mean: 0.3871089558303356
- adapter_target_condition_l1_mean: 0.3871089556316535
- pre_control_condition_l1_mean: 0.3932592959702015
- control_condition_l1_delta_mean: -0.006150340139865876
- control_payload_bytes_mean: 32.428333333333335
- control_payload_bpp_512_mean: 0.0009896341959635417
- control_grid_abs_mean: 0.3123532202964028
- condition_cosine_loss_mean: 0.20577573900421461
- condition_channel_stats_loss_mean: 0.2051039633527398
- condition_highfreq_loss_mean: 0.05054947042216857
- condition_residual_rms_guard_loss_mean: 0.011147667687256729
- condition_residual_rms_ratio_mean_mean: 0.5340722558150689
- condition_residual_rms_ratio_max_mean: 0.6577647433429956
- detail_contrast_loss_mean: 0.00030485048812503617
- detail_condition_l1_zero_mean: 0.3931912149488926
- detail_condition_l1_gap_mean: 0.006082259317239125
- detail_highfreq_residual_loss_mean: 0.28805541617174946
- detail_highfreq_residual_pred_l1_mean: 0.024510975336500755
- detail_highfreq_residual_target_l1_mean: 0.2887607687835892
- detail_residual_target_loss_mean: 0.4918909715861082
- detail_residual_target_pred_l1_mean: 0.057440479832390944
- detail_residual_target_target_l1_mean: 0.5028221279382705
- base_condition_l1_mean: 0.5179553906122843
- condition_l1_delta_vs_base_mean: -0.1308464347819487
- condition_cosine_mean: 0.7942242609957854
- pred_condition_std_mean: 0.7653582163651784
- target_condition_std_mean: 0.8107022595405579
- pred_condition_highfreq_mean: 0.23216845721006393
- target_condition_highfreq_mean: 0.24471498607347408
- image_l1_mean: 0.058646371932700274
- lpips_mean: 0.3425144420315822
- dists_loss_mean: 0.2327220623070995
- ms_ssim_loss_mean: 0.2798032117386659
- stage3_l1_guard_mean: 0.004683858180845467
- stage3_mse_guard_mean: 0.0016138394702284132
- stage4_psnr_mean: 21.168956486384072
- stage3_psnr_mean: 21.996788676579794
- condition_residual_l1_mean: 0.36409879197676975
- condition_delta_raw_l1_mean: 0.5472111593435208
- semantic_latent_drop_fraction_mean: 0.205
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage5_controlaware_adapter_lora0156_all_k16_fixedtarget_ft600_b4ga2.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage5_controlaware_adapter_lora0156_all_k16_fixedtarget_ft600_b4ga2/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage5_controlaware_adapter_lora0156_all_k16_fixedtarget_ft600_b4ga2`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_215528-5t0j0xsl`
