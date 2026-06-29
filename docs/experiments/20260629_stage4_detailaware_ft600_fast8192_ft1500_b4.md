# 20260629_stage4_detailaware_ft600_fast8192_ft1500_b4

Date: 2026-06-29T01:29:57

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage4_detailaware_ft600_fast8192_ft1500_b4 --crop-size 512 --batch-size 4 --grad-accum-steps 1 --num-workers 2 --max-steps 1500 --lr 1e-5 --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_detailaware_adapter_idstart_ft600_b2.pt --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 0.80 --condition-cosine-weight 0.25 --condition-channel-stats-weight 0.20 --condition-highfreq-weight 0.05 --image-l1-weight 0.08 --lpips-weight 0.03 --ms-ssim-weight 0.08 --stage3-l1-guard-weight 0.60 --stage3-mse-guard-weight 1.20 --grad-clip-norm 1.0 --save-sample-every 500 --wandb-mode offline
```

## Summary

- loss_mean: 0.4715002119938532
- condition_l1_mean: 0.399578200062116
- condition_cosine_loss_mean: 0.22500257066885632
- condition_channel_stats_loss_mean: 0.24911580447355905
- condition_highfreq_loss_mean: 0.05571152965476116
- base_condition_l1_mean: 0.5172687310775121
- condition_l1_delta_vs_base_mean: -0.11769053101539612
- condition_cosine_mean: 0.7749974293311437
- pred_condition_std_mean: 0.7446395278374354
- target_condition_std_mean: 0.8104142940839132
- pred_condition_highfreq_mean: 0.2263548261920611
- target_condition_highfreq_mean: 0.24459989674886068
- image_l1_mean: 0.05811491445700327
- lpips_mean: 0.40813551236192386
- dists_loss_mean: 0.0
- ms_ssim_loss_mean: 0.2763326306740443
- stage3_l1_guard_mean: 0.003961084317415953
- stage3_mse_guard_mean: 0.0013347892792274555
- stage4_psnr_mean: 21.208635848999023
- stage3_psnr_mean: 21.950961861928302
- condition_residual_l1_mean: 0.3398466455141703
- condition_delta_raw_l1_mean: 0.531188580373923

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailaware_ft600_fast8192_ft1500_b4.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage4_detailaware_ft600_fast8192_ft1500_b4/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage4_detailaware_ft600_fast8192_ft1500_b4`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_012321-gybdt1vk`
