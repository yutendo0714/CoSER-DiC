# 20260628_stage4_cod_lite_pyramid_probe50_full512_bs1

Date: 2026-06-28T17:19:25

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_gencodec512_no_pp_recon_export_prefixsafe/reconstructions/manifest.jsonl --run-name 20260628_stage4_cod_lite_pyramid_probe50_full512_bs1 --crop-size 512 --batch-size 1 --num-workers 2 --max-steps 50 --lr 2e-4 --condition-l1-weight 1.0 --image-l1-weight 0.10 --lpips-weight 0.02 --base-condition native_stage3 --adapter-kind pyramid --hidden-channels 192 --semantic-channels 3 --save-sample-every 25 --wandb-mode offline
```

## Summary

- loss_mean: 0.5123223954439163
- condition_l1_mean: 0.4950213497877121
- image_l1_mean: 0.07255147092044353
- lpips_mean: 0.5022949588298797
- stage4_psnr_mean: 20.077435550689696
- stage3_psnr_mean: 21.434522953033447

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_probe50_full512_bs1.pt`
- summary: `results/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_probe50_full512_bs1/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_probe50_full512_bs1`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_171909-s545004i`
