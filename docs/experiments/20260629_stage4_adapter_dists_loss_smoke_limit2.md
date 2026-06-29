# 20260629_stage4_adapter_dists_loss_smoke_limit2

Date: 2026-06-29T00:37:39

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_traincache_clic_oi_2048_seed20260628/reconstructions/manifest.jsonl --run-name 20260629_stage4_adapter_dists_loss_smoke_limit2 --crop-size 512 --limit 2 --batch-size 1 --num-workers 0 --max-steps 1 --lr 1e-5 --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 64 --num-image-blocks 1 --num-condition-blocks 1 --num-detail-blocks 1 --num-fusion-blocks 1 --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 0.1 --image-l1-weight 0.01 --lpips-weight 0.01 --dists-weight 0.01 --ms-ssim-weight 0.01 --stage3-l1-guard-weight 0.1 --stage3-mse-guard-weight 0.1 --wandb-mode offline --save-sample-every 0
```

## Summary

- loss_mean: 0.0685502216219902
- condition_l1_mean: 0.5197501182556152
- condition_cosine_loss_mean: 0.0
- condition_channel_stats_loss_mean: 0.0
- condition_highfreq_loss_mean: 0.0
- base_condition_l1_mean: 0.5197501182556152
- condition_l1_delta_vs_base_mean: 0.0
- condition_cosine_mean: 0.6059795618057251
- pred_condition_std_mean: 0.770107090473175
- target_condition_std_mean: 0.7331730723381042
- pred_condition_highfreq_mean: 0.316359281539917
- target_condition_highfreq_mean: 0.3041095435619354
- image_l1_mean: 0.08916924893856049
- lpips_mean: 0.6864980459213257
- dists_loss_mean: 0.38200944662094116
- ms_ssim_loss_mean: 0.3973150849342346
- stage3_l1_guard_mean: 0.008043237030506134
- stage3_mse_guard_mean: 0.002209750935435295
- stage4_psnr_mean: 17.762910842895508
- stage3_psnr_mean: 18.377803802490234
- condition_residual_l1_mean: 0.0
- condition_delta_raw_l1_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage4_adapter_dists_loss_smoke_limit2.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage4_adapter_dists_loss_smoke_limit2/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage4_adapter_dists_loss_smoke_limit2`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_003736-jin3ok1n`
