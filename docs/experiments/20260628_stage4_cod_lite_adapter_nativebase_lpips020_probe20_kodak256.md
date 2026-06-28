# 20260628_stage4_cod_lite_adapter_nativebase_lpips020_probe20_kodak256

Date: 2026-06-28T16:13:12

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_gencodec512_no_pp_recon_export_prefixsafe/reconstructions/manifest.jsonl --run-name 20260628_stage4_cod_lite_adapter_nativebase_lpips020_probe20_kodak256 --crop-size 256 --limit 24 --batch-size 1 --num-workers 2 --max-steps 20 --save-sample-every 10 --wandb-mode offline --base-condition native_stage3 --lr 1e-4 --condition-l1-weight 0.5 --image-l1-weight 0.25 --lpips-weight 0.2
```

## Summary

- loss_mean: 0.43709894716739656
- condition_l1_mean: 0.570447689294815
- image_l1_mean: 0.07368860431015492
- lpips_mean: 0.6672647267580032
- stage4_psnr_mean: 19.75267071723938
- stage3_psnr_mean: 20.700235748291014

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_adapter_nativebase_lpips020_probe20_kodak256.pt`
- summary: `results/stage4_cod_lite_adapter/20260628_stage4_cod_lite_adapter_nativebase_lpips020_probe20_kodak256/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260628_stage4_cod_lite_adapter_nativebase_lpips020_probe20_kodak256`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_161309-3jz0ogmx`
