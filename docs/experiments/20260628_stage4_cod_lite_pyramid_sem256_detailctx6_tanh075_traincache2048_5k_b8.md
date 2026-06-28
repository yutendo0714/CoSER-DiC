# 20260628_stage4_cod_lite_pyramid_sem256_detailctx6_tanh075_traincache2048_5k_b8

Date: 2026-06-28T20:56:10

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_traincache_clic_oi_2048_seed20260628/reconstructions/manifest.jsonl --run-name 20260628_stage4_cod_lite_pyramid_sem256_detailctx6_tanh075_traincache2048_5k_b8 --crop-size 512 --batch-size 8 --grad-accum-steps 1 --num-workers 4 --max-steps 5000 --lr 1e-4 --condition-l1-weight 1.0 --image-l1-weight 0.0 --lpips-weight 0.0 --base-condition native_stage3 --adapter-kind pyramid --hidden-channels 256 --semantic-channels 256 --detail-context residual_grid_codes --condition-residual-tanh --condition-residual-scale 0.75 --grad-clip-norm 1.0 --save-sample-every 0 --wandb-mode offline
```

## Summary

- loss_mean: 0.41117131090164183
- condition_l1_mean: 0.41117131090164183
- image_l1_mean: 0.0
- lpips_mean: 0.0
- stage4_psnr_mean: 0.0
- stage3_psnr_mean: 22.067516132354736
- condition_residual_l1_mean: 0.3019850692008622
- condition_delta_raw_l1_mean: 0.540385900551267

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_tanh075_traincache2048_5k_b8.pt`
- summary: `results/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_tanh075_traincache2048_5k_b8/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_tanh075_traincache2048_5k_b8`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_204104-98m02c5x`
