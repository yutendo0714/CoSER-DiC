# 20260629_stage5_controlaware_adapter_lora0156_all_k16_fixedtarget_dists_ft500_b4ga2

Date: 2026-06-29T22:15:08

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage5_controlaware_adapter_lora0156_all_k16_fixedtarget_dists_ft500_b4ga2 --crop-size 512 --batch-size 4 --grad-accum-steps 2 --num-workers 4 --max-steps 500 --lr 6.0e-7 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage5_controlaware_adapter_lora0156_all_k16_fixedtarget_ft600_b4ga2.pt --init-nonstrict --cod-lite-checkpoint external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.pt --cod-lite-config external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.yaml --cod-lite-repo external/repos/GenCodec/CoD_Lite --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --detail-control-branch --detail-film-modulation --semantic-latent-dropout-prob 0.2 --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 0.40 --condition-cosine-weight 0.14 --condition-channel-stats-weight 0.16 --condition-highfreq-weight 0.14 --condition-highfreq-threshold 0.25 --detail-contrast-weight 0.24 --detail-contrast-margin 0.003 --detail-highfreq-residual-weight 0.20 --detail-highfreq-kernel-size 5 --detail-residual-target-weight 0.08 --image-l1-weight 0.12 --lpips-weight 0.14 --dists-weight 0.36 --ms-ssim-weight 0.08 --stage3-l1-guard-weight 0.42 --stage3-mse-guard-weight 0.12 --backbone-lora-plan results/stage5_inspect/20260629_cod_lite_bpp0156_lora_target_plan_rank4.json --backbone-lora-preset denoiser_all --backbone-lora-rank 4 --backbone-lora-alpha 4.0 --backbone-lora-lr 2.5e-6 --backbone-lora-weight-decay 0.0 --train-counted-control-mode condition_residual_affine_basis --control-basis outputs/stage5_control_basis/20260629_lora0156_denoiser_all_postaffine_basis_g32s8_k256_train8192_b12/control_basis.pt --control-grid-size 8 --control-groups 32 --control-basis-components 16 --control-basis-candidate-components 256 --control-basis-selection topk --control-codec huffman --control-huffman-key topk_c256_k16_p99_b4_mulaw16 --control-quantizer mu_law --control-mu 16 --control-bits 4 --control-range 0.695025622844696 --control-affine-groups 16 --control-affine-grid-size 1 --control-affine-gain-range 1.0 --control-affine-bias-range 0.25 --control-scale 1.0 --grad-clip-norm 1.0 --wandb-mode offline --save-sample-every 250
```

## Summary

- loss_mean: 0.48326670333743094
- condition_l1_mean: 0.3868783744871616
- adapter_target_condition_l1_mean: 0.3868783746063709
- pre_control_condition_l1_mean: 0.3933409739434719
- control_condition_l1_delta_mean: -0.006462599456310272
- control_payload_bytes_mean: 32.37325
- control_payload_bpp_512_mean: 0.0009879531860351562
- control_grid_abs_mean: 0.3156292617470026
- condition_cosine_loss_mean: 0.20549887585639953
- condition_channel_stats_loss_mean: 0.20398049096763135
- condition_highfreq_loss_mean: 0.0502736091054976
- condition_residual_rms_guard_loss_mean: 0.01133920376991309
- condition_residual_rms_ratio_mean_mean: 0.5348014016747474
- condition_residual_rms_ratio_max_mean: 0.6593296033143997
- detail_contrast_loss_mean: 0.00023397389426827432
- detail_condition_l1_zero_mean: 0.39356568253040314
- detail_condition_l1_gap_mean: 0.006687307924032211
- detail_highfreq_residual_loss_mean: 0.2874844021350145
- detail_highfreq_residual_pred_l1_mean: 0.025761555030941963
- detail_highfreq_residual_target_l1_mean: 0.2883313020169735
- detail_residual_target_loss_mean: 0.48911669084429743
- detail_residual_target_pred_l1_mean: 0.06053473247587681
- detail_residual_target_target_l1_mean: 0.500847379565239
- base_condition_l1_mean: 0.5178959121108055
- condition_l1_delta_vs_base_mean: -0.13101753762364388
- condition_cosine_mean: 0.7945011241436004
- pred_condition_std_mean: 0.7652616058588028
- target_condition_std_mean: 0.8108199405670166
- pred_condition_highfreq_mean: 0.23274363331496717
- target_condition_highfreq_mean: 0.24445335932075976
- image_l1_mean: 0.05863539730757475
- lpips_mean: 0.341695123180747
- dists_loss_mean: 0.23376739728450774
- ms_ssim_loss_mean: 0.27854327058792117
- stage3_l1_guard_mean: 0.00449745079735294
- stage3_mse_guard_mean: 0.0015477914837829303
- stage4_psnr_mean: 21.18064084625244
- stage3_psnr_mean: 21.973751573562623
- condition_residual_l1_mean: 0.3637290858477354
- condition_delta_raw_l1_mean: 0.5378137492537498
- semantic_latent_drop_fraction_mean: 0.208
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage5_controlaware_adapter_lora0156_all_k16_fixedtarget_dists_ft500_b4ga2.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage5_controlaware_adapter_lora0156_all_k16_fixedtarget_dists_ft500_b4ga2/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage5_controlaware_adapter_lora0156_all_k16_fixedtarget_dists_ft500_b4ga2`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_220715-ah8u2wjg`
