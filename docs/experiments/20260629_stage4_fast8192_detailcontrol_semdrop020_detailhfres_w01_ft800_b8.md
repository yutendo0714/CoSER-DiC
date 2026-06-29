# 20260629_stage4_fast8192_detailcontrol_semdrop020_detailhfres_w01_ft800_b8

Date: 2026-06-29T02:55:12

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage4_fast8192_detailcontrol_semdrop020_detailhfres_w01_ft800_b8 --crop-size 512 --batch-size 8 --num-workers 2 --max-steps 800 --lr 5e-5 --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --detail-control-branch --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailcontrol_semdrop020_ft800_b8.pt --init-nonstrict --semantic-latent-dropout-prob 0.2 --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 1.0 --condition-cosine-weight 0.2 --condition-channel-stats-weight 0.15 --condition-highfreq-weight 0.03 --detail-highfreq-residual-weight 0.1 --detail-highfreq-kernel-size 5 --image-l1-weight 0.25 --lpips-weight 0.03 --ms-ssim-weight 0.02 --stage3-l1-guard-weight 0.2 --stage3-mse-guard-weight 0.05 --grad-clip-norm 1.0 --wandb-mode offline --save-sample-every 0
```

## Summary

- loss_mean: 0.5456625666469336
- condition_l1_mean: 0.39719586219638586
- condition_cosine_loss_mean: 0.2245921601355076
- condition_channel_stats_loss_mean: 0.2606008236296475
- condition_highfreq_loss_mean: 0.056591439112089574
- detail_contrast_loss_mean: 0.0
- detail_condition_l1_zero_mean: 0.0
- detail_condition_l1_gap_mean: 0.0
- detail_highfreq_residual_loss_mean: 0.29080817572772505
- detail_highfreq_residual_pred_l1_mean: 0.008802882128511555
- detail_highfreq_residual_target_l1_mean: 0.2907837191596627
- base_condition_l1_mean: 0.5168800438940525
- condition_l1_delta_vs_base_mean: -0.11968418169766665
- condition_cosine_mean: 0.7754078398644925
- pred_condition_std_mean: 0.7319314756989479
- target_condition_std_mean: 0.8098107255250215
- pred_condition_highfreq_mean: 0.2239516644924879
- target_condition_highfreq_mean: 0.24497095528990032
- image_l1_mean: 0.059182911519892516
- lpips_mean: 0.4058072051033378
- dists_loss_mean: 0.0
- ms_ssim_loss_mean: 0.2799307233095169
- stage3_l1_guard_mean: 0.005148746256600134
- stage3_mse_guard_mean: 0.0016255942515999778
- stage4_psnr_mean: 21.094784042835236
- stage3_psnr_mean: 21.993137352466583
- condition_residual_l1_mean: 0.3331281941756606
- condition_delta_raw_l1_mean: 0.5179907619953156
- semantic_latent_drop_fraction_mean: 0.20140625
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailcontrol_semdrop020_detailhfres_w01_ft800_b8.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailcontrol_semdrop020_detailhfres_w01_ft800_b8/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailcontrol_semdrop020_detailhfres_w01_ft800_b8`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_024744-ag0m6ibb`
