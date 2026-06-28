# 20260628_stage4_cod_lite_adapter_nativebase_smoke2_manifest128

Date: 2026-06-28T16:10:46

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_gencodec512_no_pp_recon_export_prefixsafe/reconstructions/manifest.jsonl --run-name 20260628_stage4_cod_lite_adapter_nativebase_smoke2_manifest128 --crop-size 128 --limit 2 --batch-size 1 --num-workers 0 --max-steps 2 --save-sample-every 1 --wandb-mode offline --base-condition native_stage3
```

## Summary

- loss_mean: 0.6312183141708374
- condition_l1_mean: 0.6149937212467194
- image_l1_mean: 0.06489833444356918
- stage4_psnr_mean: 21.51510524749756
- stage3_psnr_mean: 22.71565341949463

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_adapter_nativebase_smoke2_manifest128.pt`
- summary: `results/stage4_cod_lite_adapter/20260628_stage4_cod_lite_adapter_nativebase_smoke2_manifest128/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260628_stage4_cod_lite_adapter_nativebase_smoke2_manifest128`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_161045-d1p4s0u8`
