# 20260628_stage4_cod_lite_adapter_nativebase_lpips020_full512_5k

Date: 2026-06-28T16:31:47

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_gencodec512_no_pp_recon_export_prefixsafe/reconstructions/manifest.jsonl --run-name 20260628_stage4_cod_lite_adapter_nativebase_lpips020_full512_5k --crop-size 512 --limit 552 --batch-size 1 --num-workers 2 --max-steps 5000 --save-sample-every 500 --wandb-mode offline --base-condition native_stage3 --lr 1e-4 --condition-l1-weight 0.5 --image-l1-weight 0.25 --lpips-weight 0.2
```

## Summary

- loss_mean: 0.33685834283828736
- condition_l1_mean: 0.46029493736624716
- image_l1_mean: 0.06716533602289855
- lpips_mean: 0.44959769294783475
- stage4_psnr_mean: 20.313636965942383
- stage3_psnr_mean: 22.006914478683473

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_adapter_nativebase_lpips020_full512_5k.pt`
- summary: `results/stage4_cod_lite_adapter/20260628_stage4_cod_lite_adapter_nativebase_lpips020_full512_5k/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260628_stage4_cod_lite_adapter_nativebase_lpips020_full512_5k`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_162409-2t6jkn4t`
