# 20260629_stage4_fast8192_detailcontrol_smoke2_b4

Date: 2026-06-29T02:03:51

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage4_fast8192_detailcontrol_smoke2_b4 --crop-size 512 --limit 8 --batch-size 4 --num-workers 2 --max-steps 2 --lr 5e-6 --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --detail-control-branch --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailaware_ft600_fast8192_ft1500_b4.pt --init-nonstrict --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 1.0 --condition-cosine-weight 0.20 --condition-channel-stats-weight 0.15 --condition-highfreq-weight 0.03 --image-l1-weight 0.25 --lpips-weight 0.03 --ms-ssim-weight 0.02 --stage3-l1-guard-weight 0.20 --stage3-mse-guard-weight 0.05 --grad-clip-norm 1.0 --save-sample-every 1 --wandb-mode offline
```

## Summary

- loss_mean: 0.4950404465198517
- condition_l1_mean: 0.3845491260290146
- condition_cosine_loss_mean: 0.21717745065689087
- condition_channel_stats_loss_mean: 0.23424968123435974
- condition_highfreq_loss_mean: 0.05222139693796635
- base_condition_l1_mean: 0.4971630871295929
- condition_l1_delta_vs_base_mean: -0.11261396110057831
- condition_cosine_mean: 0.7828225493431091
- pred_condition_std_mean: 0.7236171662807465
- target_condition_std_mean: 0.7957602441310883
- pred_condition_highfreq_mean: 0.22854559123516083
- target_condition_highfreq_mean: 0.25236576795578003
- image_l1_mean: 0.055256208404898643
- lpips_mean: 0.3653256744146347
- dists_loss_mean: 0.0
- ms_ssim_loss_mean: 0.23680543899536133
- stage3_l1_guard_mean: 0.003926682053133845
- stage3_mse_guard_mean: 0.0011298353783786297
- stage4_psnr_mean: 21.201637268066406
- stage3_psnr_mean: 21.94315242767334
- condition_residual_l1_mean: 0.3271180987358093
- condition_delta_raw_l1_mean: 0.5002653896808624
- semantic_latent_drop_fraction_mean: 0.0
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailcontrol_smoke2_b4.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailcontrol_smoke2_b4/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailcontrol_smoke2_b4`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_020348-aougzzqx`
