# 20260628_stage4_cod_lite_adapter_lpips020_alpha010_full552_eval

Date: 2026-06-28T17:04:02

## Command

```bash
scripts/eval_stage4_cod_lite_adapter.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_adapter_nativebase_lpips020_full512_5k.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_gencodec512_no_pp_recon_export_prefixsafe/reconstructions/manifest.jsonl --per-image-metrics results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_gencodec512_no_pp_recon_export_prefixsafe/per_image_metrics.jsonl --run-name 20260628_stage4_cod_lite_adapter_lpips020_alpha010_full552_eval --crop-size 512 --limit 552 --batch-size 1 --num-workers 2 --blend-alpha 0.1 --wandb-mode offline
```

## Summary

- actual_payload_bpp_mean: 0.013999275539232336
- stage3_psnr_mean: 21.995091400284696
- stage4_psnr_mean: 22.014686000519905
- stage3_ms_ssim_mean: 0.7348259755180798
- stage4_ms_ssim_mean: 0.7364838184194936
- stage3_lpips_alex_mean: 0.5757584185541972
- stage4_lpips_alex_mean: 0.5626209777361457
- stage3_dists_mean: 0.35361975042716315
- stage4_dists_mean: 0.3529168239970138
- stage4_l1_mean: 0.055737176511868616
- stage3_l1_mean: 0.05591988528106848
- count: 552
- stage4_psnr_win_rate: 0.7626811594202898
- stage4_ms_ssim_win_rate: 0.9601449275362319
- stage4_lpips_win_rate: 0.9981884057971014
- stage4_dists_win_rate: 0.5507246376811594
- stage4_blend_alpha: 0.1
- stage4_payload_policy: inherits Stage 3 semantic/detail actual_payload_bpp; fixed CoD-Lite weights, adapter, and deterministic decoder-side blend alpha are not image-specific side information

## Artifacts

- summary: `results/stage4_cod_lite_adapter_eval/20260628_stage4_cod_lite_adapter_lpips020_alpha010_full552_eval/summary.json`
- per_image: `results/stage4_cod_lite_adapter_eval/20260628_stage4_cod_lite_adapter_lpips020_alpha010_full552_eval/per_image_metrics.jsonl`
- reconstructions: ``
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_170318-0estmham`
