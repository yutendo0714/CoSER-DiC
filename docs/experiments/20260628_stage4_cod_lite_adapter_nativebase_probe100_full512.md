# 20260628_stage4_cod_lite_adapter_nativebase_probe100_full512

Date: 2026-06-28T16:12:08

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_gencodec512_no_pp_recon_export_prefixsafe/reconstructions/manifest.jsonl --run-name 20260628_stage4_cod_lite_adapter_nativebase_probe100_full512 --crop-size 512 --limit 552 --batch-size 1 --num-workers 2 --max-steps 100 --save-sample-every 50 --wandb-mode offline --base-condition native_stage3 --lr 2e-4
```

## Summary

- loss_mean: 0.49165910214185715
- condition_l1_mean: 0.4758564680814743
- image_l1_mean: 0.06321053778752685
- stage4_psnr_mean: 20.557871761322023
- stage3_psnr_mean: 21.919245252609254

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_adapter_nativebase_probe100_full512.pt`
- summary: `results/stage4_cod_lite_adapter/20260628_stage4_cod_lite_adapter_nativebase_probe100_full512/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260628_stage4_cod_lite_adapter_nativebase_probe100_full512`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_161157-jiromvl7`
