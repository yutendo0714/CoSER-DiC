# 20260628_stage4_cod_lite_pyramid_sem256_detailctx6_tanh075_traincache2048_smoke_b8

Date: 2026-06-28T20:40:03

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_traincache_clic_oi_2048_seed20260628/reconstructions/manifest.jsonl --run-name 20260628_stage4_cod_lite_pyramid_sem256_detailctx6_tanh075_traincache2048_smoke_b8 --crop-size 512 --limit 64 --batch-size 8 --grad-accum-steps 1 --num-workers 2 --max-steps 5 --lr 1e-4 --condition-l1-weight 1.0 --image-l1-weight 0.0 --lpips-weight 0.0 --base-condition native_stage3 --adapter-kind pyramid --hidden-channels 256 --semantic-channels 256 --detail-context residual_grid_codes --condition-residual-tanh --condition-residual-scale 0.75 --grad-clip-norm 1.0 --save-sample-every 0 --wandb-mode offline
```

## Summary

- loss_mean: 0.518498158454895
- condition_l1_mean: 0.518498158454895
- image_l1_mean: 0.0
- lpips_mean: 0.0
- stage4_psnr_mean: 0.0
- stage3_psnr_mean: 22.005429458618163
- condition_residual_l1_mean: 0.012158088944852352
- condition_delta_raw_l1_mean: 0.016226624697446825

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_tanh075_traincache2048_smoke_b8.pt`
- summary: `results/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_tanh075_traincache2048_smoke_b8/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_tanh075_traincache2048_smoke_b8`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_204001-o8jerpbf`
