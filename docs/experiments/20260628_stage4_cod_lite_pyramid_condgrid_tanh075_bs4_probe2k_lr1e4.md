# 20260628_stage4_cod_lite_pyramid_condgrid_tanh075_bs4_probe2k_lr1e4

Date: 2026-06-28T17:49:51

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_gencodec512_no_pp_recon_export_prefixsafe/reconstructions/manifest.jsonl --run-name 20260628_stage4_cod_lite_pyramid_condgrid_tanh075_bs4_probe2k_lr1e4 --crop-size 512 --batch-size 4 --num-workers 4 --max-steps 2000 --lr 1e-4 --condition-l1-weight 1.0 --image-l1-weight 0.0 --lpips-weight 0.0 --base-condition native_stage3 --adapter-kind pyramid --hidden-channels 256 --semantic-channels 3 --condition-residual-tanh --condition-residual-scale 0.75 --grad-clip-norm 1.0 --save-sample-every 0 --wandb-mode offline
```

## Summary

- loss_mean: 0.43853643487393856
- condition_l1_mean: 0.43853643487393856
- image_l1_mean: 0.0
- lpips_mean: 0.0
- stage4_psnr_mean: 0.0
- stage3_psnr_mean: 21.99477844810486
- condition_residual_l1_mean: 0.29631257505188113
- condition_delta_raw_l1_mean: 0.5404700995978201

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_condgrid_tanh075_bs4_probe2k_lr1e4.pt`
- summary: `results/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_condgrid_tanh075_bs4_probe2k_lr1e4/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_condgrid_tanh075_bs4_probe2k_lr1e4`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_174427-3n3oopa0`
