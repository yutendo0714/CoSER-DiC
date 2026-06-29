# 20260629_stage4_detailaware_ft600_fast8192_b4_smoke1

Date: 2026-06-29T01:23:03

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage4_detailaware_ft600_fast8192_b4_smoke1 --crop-size 512 --batch-size 4 --grad-accum-steps 1 --num-workers 2 --max-steps 1 --lr 1e-5 --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_detailaware_adapter_idstart_ft600_b2.pt --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 0.80 --condition-cosine-weight 0.25 --condition-channel-stats-weight 0.20 --condition-highfreq-weight 0.05 --image-l1-weight 0.08 --lpips-weight 0.03 --ms-ssim-weight 0.08 --stage3-l1-guard-weight 0.60 --stage3-mse-guard-weight 1.20 --grad-clip-norm 1.0 --save-sample-every 0 --wandb-mode offline
```

## Summary

- loss_mean: 0.4944956600666046
- condition_l1_mean: 0.40601253509521484
- condition_cosine_loss_mean: 0.2598620653152466
- condition_channel_stats_loss_mean: 0.2594391107559204
- condition_highfreq_loss_mean: 0.06421777606010437
- base_condition_l1_mean: 0.4977138936519623
- condition_l1_delta_vs_base_mean: -0.09170135855674744
- condition_cosine_mean: 0.7401379346847534
- pred_condition_std_mean: 0.6939186453819275
- target_condition_std_mean: 0.7573367357254028
- pred_condition_highfreq_mean: 0.24508200585842133
- target_condition_highfreq_mean: 0.2775115370750427
- image_l1_mean: 0.0576445572078228
- lpips_mean: 0.40619662404060364
- dists_loss_mean: 0.0
- ms_ssim_loss_mean: 0.3257637619972229
- stage3_l1_guard_mean: 0.007556716911494732
- stage3_mse_guard_mean: 0.001857363386079669
- stage4_psnr_mean: 21.499881744384766
- stage3_psnr_mean: 22.623947143554688
- condition_residual_l1_mean: 0.35027146339416504
- condition_delta_raw_l1_mean: 0.5492454767227173

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailaware_ft600_fast8192_b4_smoke1.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage4_detailaware_ft600_fast8192_b4_smoke1/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage4_detailaware_ft600_fast8192_b4_smoke1`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_012300-wzp6se1t`
