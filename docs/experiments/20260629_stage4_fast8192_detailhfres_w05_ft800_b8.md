# 20260629_stage4_fast8192_detailhfres_w05_ft800_b8

Date: 2026-06-29T02:45:58

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage4_fast8192_detailhfres_w05_ft800_b8 --crop-size 512 --batch-size 8 --num-workers 2 --max-steps 800 --lr 5e-5 --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --detail-control-branch --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailcontrol_ft800_b4.pt --init-nonstrict --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 1.0 --condition-cosine-weight 0.2 --condition-channel-stats-weight 0.15 --condition-highfreq-weight 0.03 --detail-highfreq-residual-weight 0.5 --detail-highfreq-kernel-size 5 --image-l1-weight 0.25 --lpips-weight 0.03 --ms-ssim-weight 0.02 --stage3-l1-guard-weight 0.2 --stage3-mse-guard-weight 0.05 --grad-clip-norm 1.0 --wandb-mode offline --save-sample-every 0
```

## Summary

- loss_mean: 0.659821068495512
- condition_l1_mean: 0.39662323918193576
- condition_cosine_loss_mean: 0.22368783958256244
- condition_channel_stats_loss_mean: 0.25916476733982563
- condition_highfreq_loss_mean: 0.05671356708277017
- detail_contrast_loss_mean: 0.0
- detail_condition_l1_zero_mean: 0.0
- detail_condition_l1_gap_mean: 0.0
- detail_highfreq_residual_loss_mean: 0.28947434458881616
- detail_highfreq_residual_pred_l1_mean: 0.013553245436633006
- detail_highfreq_residual_target_l1_mean: 0.2907837191596627
- base_condition_l1_mean: 0.5168800438940525
- condition_l1_delta_vs_base_mean: -0.12025680471211672
- condition_cosine_mean: 0.7763121604174376
- pred_condition_std_mean: 0.7337066478282214
- target_condition_std_mean: 0.8098107255250215
- pred_condition_highfreq_mean: 0.22282948829233645
- target_condition_highfreq_mean: 0.24497095528990032
- image_l1_mean: 0.05815759481862187
- lpips_mean: 0.404958944991231
- dists_loss_mean: 0.0
- ms_ssim_loss_mean: 0.27784456215798853
- stage3_l1_guard_mean: 0.004150358900951687
- stage3_mse_guard_mean: 0.0014367878516350174
- stage4_psnr_mean: 21.1977658700943
- stage3_psnr_mean: 21.993137352466583
- condition_residual_l1_mean: 0.33410336650907996
- condition_delta_raw_l1_mean: 0.5187527187541128
- semantic_latent_drop_fraction_mean: 0.0
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailhfres_w05_ft800_b8.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailhfres_w05_ft800_b8/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailhfres_w05_ft800_b8`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_023830-dhebebv8`
