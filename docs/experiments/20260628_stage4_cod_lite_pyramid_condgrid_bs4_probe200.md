# 20260628_stage4_cod_lite_pyramid_condgrid_bs4_probe200

Date: 2026-06-28T17:26:34

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_gencodec512_no_pp_recon_export_prefixsafe/reconstructions/manifest.jsonl --run-name 20260628_stage4_cod_lite_pyramid_condgrid_bs4_probe200 --crop-size 512 --batch-size 4 --num-workers 4 --max-steps 200 --lr 3e-4 --condition-l1-weight 1.0 --image-l1-weight 0.0 --lpips-weight 0.0 --base-condition native_stage3 --adapter-kind pyramid --hidden-channels 256 --semantic-channels 3 --save-sample-every 0 --wandb-mode offline
```

## Summary

- loss_mean: 0.8614138552546501
- condition_l1_mean: 0.8614138552546501
- image_l1_mean: 0.0
- lpips_mean: 0.0
- stage4_psnr_mean: 0.0
- stage3_psnr_mean: 22.01956151008606

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_condgrid_bs4_probe200.pt`
- summary: `results/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_condgrid_bs4_probe200/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_condgrid_bs4_probe200`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_172600-7e7t4lxa`
