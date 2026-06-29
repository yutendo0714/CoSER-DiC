# 20260629_stage5_detailcontrol_hfcontext_lora0156_all_k16_ft500_b4ga2

Date: 2026-06-29T23:02:08

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage5_detailcontrol_hfcontext_lora0156_all_k16_ft500_b4ga2 --crop-size 512 --batch-size 4 --grad-accum-steps 2 --num-workers 4 --max-steps 500 --lr 1.0e-6 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage5_detailtarget_lora0156_denoiser_all_r4_balanced_ft900_b4ga2.pt --init-nonstrict --cod-lite-checkpoint external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.pt --cod-lite-config external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.yaml --cod-lite-repo external/repos/GenCodec/CoD_Lite --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --detail-control-branch --detail-control-blocks 2 --detail-control-condition-fusion --detail-highfreq-context-branch --detail-film-modulation --semantic-latent-dropout-prob 0.2 --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 0.44 --condition-cosine-weight 0.16 --condition-channel-stats-weight 0.16 --condition-highfreq-weight 0.12 --condition-highfreq-threshold 0.25 --detail-contrast-weight 0.24 --detail-contrast-margin 0.003 --detail-highfreq-residual-weight 0.20 --detail-highfreq-kernel-size 5 --detail-residual-target-weight 0.08 --image-l1-weight 0.14 --lpips-weight 0.18 --dists-weight 0.26 --ms-ssim-weight 0.08 --stage3-l1-guard-weight 0.45 --stage3-mse-guard-weight 0.12 --backbone-lora-plan results/stage5_inspect/20260629_cod_lite_bpp0156_lora_target_plan_rank4.json --backbone-lora-preset denoiser_all --backbone-lora-rank 4 --backbone-lora-alpha 4.0 --backbone-lora-lr 4.0e-6 --backbone-lora-weight-decay 0.0 --train-counted-control-mode condition_residual_affine_basis --control-basis outputs/stage5_control_basis/20260629_lora0156_denoiser_all_postaffine_basis_g32s8_k256_train8192_b12/control_basis.pt --control-grid-size 8 --control-groups 32 --control-basis-components 16 --control-basis-candidate-components 256 --control-basis-selection topk --control-codec huffman --control-huffman-key topk_c256_k16_p99_b4_mulaw16 --control-quantizer mu_law --control-mu 16 --control-bits 4 --control-range 0.695025622844696 --control-affine-groups 16 --control-affine-grid-size 1 --control-affine-gain-range 1.0 --control-affine-bias-range 0.25 --control-scale 1.0 --grad-clip-norm 1.0 --wandb-mode offline --save-sample-every 250
```

## Summary

- loss_mean: 0.4954817863702774
- condition_l1_mean: 0.38779332545399664
- adapter_target_condition_l1_mean: 0.3877933252155781
- pre_control_condition_l1_mean: 0.39392219644784926
- control_condition_l1_delta_mean: -0.0061288709938526155
- control_payload_bytes_mean: 32.42675
- control_payload_bpp_512_mean: 0.0009895858764648437
- control_grid_abs_mean: 0.31284344479441645
- condition_cosine_loss_mean: 0.20741766107082366
- condition_channel_stats_loss_mean: 0.20603956711292268
- condition_highfreq_loss_mean: 0.05070031899213791
- condition_residual_rms_guard_loss_mean: 0.011568056772954747
- condition_residual_rms_ratio_mean_mean: 0.536084826618433
- condition_residual_rms_ratio_max_mean: 0.6591958878934383
- detail_contrast_loss_mean: 0.0002618784075602889
- detail_condition_l1_zero_mean: 0.39394482311606405
- detail_condition_l1_gap_mean: 0.006151497900485993
- detail_highfreq_residual_loss_mean: 0.2880584863126278
- detail_highfreq_residual_pred_l1_mean: 0.02477848288603127
- detail_highfreq_residual_target_l1_mean: 0.2887683515250683
- detail_residual_target_loss_mean: 0.49221251165866853
- detail_residual_target_pred_l1_mean: 0.05816048670187592
- detail_residual_target_target_l1_mean: 0.5033819377720355
- base_condition_l1_mean: 0.5188397481739521
- condition_l1_delta_vs_base_mean: -0.13104642271995545
- condition_cosine_mean: 0.7925823389291763
- pred_condition_std_mean: 0.7641530198454857
- target_condition_std_mean: 0.8095126668810845
- pred_condition_highfreq_mean: 0.23216222621500493
- target_condition_highfreq_mean: 0.24450671049952508
- image_l1_mean: 0.059149410773068664
- lpips_mean: 0.3434438520371914
- dists_loss_mean: 0.23312667770683765
- ms_ssim_loss_mean: 0.28171215587854387
- stage3_l1_guard_mean: 0.004703346927883103
- stage3_mse_guard_mean: 0.0016396484769647941
- stage4_psnr_mean: 21.105169616699218
- stage3_psnr_mean: 21.934375907897948
- condition_residual_l1_mean: 0.3647048828750849
- condition_delta_raw_l1_mean: 0.5477733287513256
- semantic_latent_drop_fraction_mean: 0.208
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage5_detailcontrol_hfcontext_lora0156_all_k16_ft500_b4ga2.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage5_detailcontrol_hfcontext_lora0156_all_k16_ft500_b4ga2/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage5_detailcontrol_hfcontext_lora0156_all_k16_ft500_b4ga2`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_225410-gmmhlrxv`
