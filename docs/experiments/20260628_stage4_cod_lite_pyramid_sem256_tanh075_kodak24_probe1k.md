# 20260628_stage4_cod_lite_pyramid_sem256_tanh075_kodak24_probe1k

Date: 2026-06-28T17:57:59

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_kodak24_feature_cache_smoke/reconstructions/manifest.jsonl --run-name 20260628_stage4_cod_lite_pyramid_sem256_tanh075_kodak24_probe1k --crop-size 512 --batch-size 4 --num-workers 4 --max-steps 1000 --lr 1e-4 --condition-l1-weight 1.0 --image-l1-weight 0.0 --lpips-weight 0.0 --base-condition native_stage3 --adapter-kind pyramid --hidden-channels 256 --semantic-channels 256 --condition-residual-tanh --condition-residual-scale 0.75 --grad-clip-norm 1.0 --save-sample-every 0 --wandb-mode offline
```

## Summary

- loss_mean: 0.4164084107875824
- condition_l1_mean: 0.4164084107875824
- image_l1_mean: 0.0
- lpips_mean: 0.0
- stage4_psnr_mean: 0.0
- stage3_psnr_mean: 21.666236392974852
- condition_residual_l1_mean: 0.3511779604605399
- condition_delta_raw_l1_mean: 0.7394140490824357

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_tanh075_kodak24_probe1k.pt`
- summary: `results/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_tanh075_kodak24_probe1k/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_tanh075_kodak24_probe1k`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_175541-5udu1963`
