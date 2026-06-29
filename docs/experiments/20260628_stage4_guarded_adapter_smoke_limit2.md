# 20260628_stage4_guarded_adapter_smoke_limit2

Date: 2026-06-28T23:28:46

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_traincache_clic_oi_2048_seed20260628/reconstructions/manifest.jsonl --run-name 20260628_stage4_guarded_adapter_smoke_limit2 --crop-size 512 --limit 2 --batch-size 1 --grad-accum-steps 1 --num-workers 0 --max-steps 1 --lr 1e-4 --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 64 --condition-residual-scale 0.5 --condition-residual-tanh --condition-l1-weight 1.0 --condition-cosine-weight 0.1 --condition-channel-stats-weight 0.05 --condition-highfreq-weight 0.02 --image-l1-weight 0.1 --ms-ssim-weight 0.1 --stage3-l1-guard-weight 0.5 --stage3-mse-guard-weight 1.0 --grad-clip-norm 1.0 --save-sample-every 0 --wandb-mode offline
```

## Summary

- loss_mean: 0.6334437131881714
- condition_l1_mean: 0.5197501182556152
- condition_cosine_loss_mean: 0.3940204381942749
- condition_channel_stats_loss_mean: 0.3677545189857483
- condition_highfreq_loss_mean: 0.05119934305548668
- base_condition_l1_mean: 0.5197501182556152
- condition_l1_delta_vs_base_mean: 0.0
- condition_cosine_mean: 0.6059795618057251
- pred_condition_std_mean: 0.770107090473175
- target_condition_std_mean: 0.7331730723381042
- pred_condition_highfreq_mean: 0.316359281539917
- target_condition_highfreq_mean: 0.3041095435619354
- image_l1_mean: 0.08916924893856049
- lpips_mean: 0.0
- ms_ssim_loss_mean: 0.3973150849342346
- stage3_l1_guard_mean: 0.008043237030506134
- stage3_mse_guard_mean: 0.002209750935435295
- stage4_psnr_mean: 17.762910842895508
- stage3_psnr_mean: 18.377803802490234
- condition_residual_l1_mean: 0.0
- condition_delta_raw_l1_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260628_stage4_guarded_adapter_smoke_limit2.pt`
- summary: `results/stage4_cod_lite_adapter/20260628_stage4_guarded_adapter_smoke_limit2/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260628_stage4_guarded_adapter_smoke_limit2`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_232845-he6v444b`
