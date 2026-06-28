# 20260628_stage4_cod_lite_adapter_nativebase_lpips005_l1guard_full512_5k

Date: 2026-06-28T16:43:29

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_gencodec512_no_pp_recon_export_prefixsafe/reconstructions/manifest.jsonl --run-name 20260628_stage4_cod_lite_adapter_nativebase_lpips005_l1guard_full512_5k --crop-size 512 --limit 552 --batch-size 1 --num-workers 2 --max-steps 5000 --save-sample-every 500 --wandb-mode offline --base-condition native_stage3 --lr 5e-5 --condition-l1-weight 1.0 --image-l1-weight 1.0 --lpips-weight 0.05
```

## Summary

- loss_mean: 0.5485078388035297
- condition_l1_mean: 0.46074918894767763
- image_l1_mean: 0.0643693338830024
- lpips_mean: 0.4677863496661186
- stage4_psnr_mean: 20.70888563709259
- stage3_psnr_mean: 22.006914478683473

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_adapter_nativebase_lpips005_l1guard_full512_5k.pt`
- summary: `results/stage4_cod_lite_adapter/20260628_stage4_cod_lite_adapter_nativebase_lpips005_l1guard_full512_5k/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260628_stage4_cod_lite_adapter_nativebase_lpips005_l1guard_full512_5k`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_163549-xgjt0dcc`
