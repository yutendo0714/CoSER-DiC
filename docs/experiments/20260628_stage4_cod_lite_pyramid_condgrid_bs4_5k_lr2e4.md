# 20260628_stage4_cod_lite_pyramid_condgrid_bs4_5k_lr2e4

Date: 2026-06-28T17:40:34

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_gencodec512_no_pp_recon_export_prefixsafe/reconstructions/manifest.jsonl --run-name 20260628_stage4_cod_lite_pyramid_condgrid_bs4_5k_lr2e4 --crop-size 512 --batch-size 4 --num-workers 4 --max-steps 5000 --lr 2e-4 --condition-l1-weight 1.0 --image-l1-weight 0.0 --lpips-weight 0.0 --base-condition native_stage3 --adapter-kind pyramid --hidden-channels 256 --semantic-channels 3 --save-sample-every 0 --wandb-mode offline
```

## Summary

- loss_mean: 40.94191009270549
- condition_l1_mean: 40.94191009270549
- image_l1_mean: 0.0
- lpips_mean: 0.0
- stage4_psnr_mean: 0.0
- stage3_psnr_mean: 21.9950188495636

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_condgrid_bs4_5k_lr2e4.pt`
- summary: `results/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_condgrid_bs4_5k_lr2e4/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_condgrid_bs4_5k_lr2e4`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_172710-nl6zt8gd`
