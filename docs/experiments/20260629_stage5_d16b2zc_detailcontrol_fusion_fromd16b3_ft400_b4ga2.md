# 20260629_stage5_d16b2zc_detailcontrol_fusion_fromd16b3_ft400_b4ga2

Date: 2026-06-30T00:18:26

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_2048_seed20260629_d16_b2_zc/reconstructions/manifest.jsonl --run-name 20260629_stage5_d16b2zc_detailcontrol_fusion_fromd16b3_ft400_b4ga2 --crop-size 512 --batch-size 4 --grad-accum-steps 2 --num-workers 4 --max-steps 400 --lr 1.0e-6 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage5_d16b3_detailcontrol_fusion_ft400_b4ga2.pt --init-nonstrict --cod-lite-checkpoint external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.pt --cod-lite-config external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.yaml --cod-lite-repo external/repos/GenCodec/CoD_Lite --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --detail-control-branch --detail-control-blocks 2 --detail-control-condition-fusion --detail-film-modulation --semantic-latent-dropout-prob 0.2 --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 0.44 --condition-cosine-weight 0.16 --condition-channel-stats-weight 0.16 --condition-highfreq-weight 0.12 --condition-highfreq-threshold 0.25 --detail-contrast-weight 0.24 --detail-contrast-margin 0.003 --detail-highfreq-residual-weight 0.18 --detail-highfreq-kernel-size 5 --detail-residual-target-weight 0.06 --image-l1-weight 0.14 --lpips-weight 0.18 --dists-weight 0.24 --ms-ssim-weight 0.08 --stage3-l1-guard-weight 0.45 --stage3-mse-guard-weight 0.12 --backbone-lora-plan results/stage5_inspect/20260629_cod_lite_bpp0156_lora_target_plan_rank4.json --backbone-lora-preset denoiser_all --backbone-lora-rank 4 --backbone-lora-alpha 4.0 --backbone-lora-lr 4.0e-6 --backbone-lora-weight-decay 0.0 --train-counted-control-mode condition_residual_affine_basis --control-basis outputs/stage5_control_basis/20260629_lora0156_denoiser_all_postaffine_basis_g32s8_k256_train8192_b12/control_basis.pt --control-grid-size 8 --control-groups 32 --control-basis-components 16 --control-basis-candidate-components 256 --control-basis-selection topk --control-codec huffman --control-huffman-key topk_c256_k16_p99_b4_mulaw16 --control-quantizer mu_law --control-mu 16 --control-bits 4 --control-range 0.695025622844696 --control-affine-groups 16 --control-affine-grid-size 1 --control-affine-gain-range 1.0 --control-affine-bias-range 0.25 --control-scale 1.0 --grad-clip-norm 1.0 --wandb-mode offline --save-sample-every 200
```

## Summary

- loss_mean: 0.48784426238387824
- condition_l1_mean: 0.3971022893860936
- adapter_target_condition_l1_mean: 0.3971022890508175
- pre_control_condition_l1_mean: 0.4043686977028847
- control_condition_l1_delta_mean: -0.007266408316791057
- control_payload_bytes_mean: 32.2346875
- control_payload_bpp_512_mean: 0.000983724594116211
- control_grid_abs_mean: 0.3303499086201191
- condition_cosine_loss_mean: 0.2168894138932228
- condition_channel_stats_loss_mean: 0.21659543953835964
- condition_highfreq_loss_mean: 0.05175848845858127
- condition_residual_rms_guard_loss_mean: 0.009257711753832575
- condition_residual_rms_ratio_mean_mean: 0.5177143842726946
- condition_residual_rms_ratio_max_mean: 0.6411377447843551
- detail_contrast_loss_mean: 0.0025369796075392515
- detail_condition_l1_zero_mean: 0.39765550568699837
- detail_condition_l1_gap_mean: 0.0005532166361808777
- detail_highfreq_residual_loss_mean: 0.29303021935746076
- detail_highfreq_residual_pred_l1_mean: 0.012539992987876758
- detail_highfreq_residual_target_l1_mean: 0.29213132154196503
- detail_residual_target_loss_mean: 0.515464025363326
- detail_residual_target_pred_l1_mean: 0.021295416426146404
- detail_residual_target_target_l1_mean: 0.5146191177144647
- base_condition_l1_mean: 0.5184862609952688
- condition_l1_delta_vs_base_mean: -0.1213839716091752
- condition_cosine_mean: 0.7831105861067772
- pred_condition_std_mean: 0.7673330669850111
- target_condition_std_mean: 0.8086724028736353
- pred_condition_highfreq_mean: 0.2369293134100735
- target_condition_highfreq_mean: 0.24553607540205122
- image_l1_mean: 0.05980810628272593
- lpips_mean: 0.35369087938219307
- dists_loss_mean: 0.23614385994151235
- ms_ssim_loss_mean: 0.277453395575285
- stage3_l1_guard_mean: 0.004807138330361341
- stage3_mse_guard_mean: 0.001641678494743246
- stage4_psnr_mean: 21.08354946374893
- stage3_psnr_mean: 21.91064417362213
- condition_residual_l1_mean: 0.352195122204721
- condition_delta_raw_l1_mean: 0.5575467803701758
- semantic_latent_drop_fraction_mean: 0.21125
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage5_d16b2zc_detailcontrol_fusion_fromd16b3_ft400_b4ga2.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage5_d16b2zc_detailcontrol_fusion_fromd16b3_ft400_b4ga2/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage5_d16b2zc_detailcontrol_fusion_fromd16b3_ft400_b4ga2`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260630_001205-0jwbqwwe`
