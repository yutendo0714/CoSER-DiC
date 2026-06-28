# 20260628_stage4_cod_lite_adapter_lpips020_alpha010_kodak24_eval

Date: 2026-06-28T17:02:34

## Command

```bash
scripts/eval_stage4_cod_lite_adapter.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_adapter_nativebase_lpips020_full512_5k.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_gencodec512_no_pp_recon_export_prefixsafe/reconstructions/manifest.jsonl --per-image-metrics results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_gencodec512_no_pp_recon_export_prefixsafe/per_image_metrics.jsonl --run-name 20260628_stage4_cod_lite_adapter_lpips020_alpha010_kodak24_eval --crop-size 512 --limit 24 --batch-size 1 --num-workers 2 --blend-alpha 0.1 --save-reconstructions --save-reconstruction-limit 24 --wandb-mode offline
```

## Summary

- actual_payload_bpp_mean: 0.014120737711588541
- stage3_psnr_mean: 21.667193094889324
- stage4_psnr_mean: 21.67846155166626
- stage3_ms_ssim_mean: 0.7225408007701238
- stage4_ms_ssim_mean: 0.7241999631126722
- stage3_lpips_alex_mean: 0.6366225394109885
- stage4_lpips_alex_mean: 0.6203984978298346
- stage3_dists_mean: 0.37316765636205673
- stage4_dists_mean: 0.3725142329931259
- stage4_l1_mean: 0.05418088256071011
- stage3_l1_mean: 0.05429107012848059
- count: 24
- stage4_psnr_win_rate: 0.7916666666666666
- stage4_ms_ssim_win_rate: 1.0
- stage4_lpips_win_rate: 1.0
- stage4_dists_win_rate: 0.5
- stage4_blend_alpha: 0.1
- stage4_payload_policy: inherits Stage 3 semantic/detail actual_payload_bpp; fixed CoD-Lite weights, adapter, and deterministic decoder-side blend alpha are not image-specific side information

## Artifacts

- summary: `results/stage4_cod_lite_adapter_eval/20260628_stage4_cod_lite_adapter_lpips020_alpha010_kodak24_eval/summary.json`
- per_image: `results/stage4_cod_lite_adapter_eval/20260628_stage4_cod_lite_adapter_lpips020_alpha010_kodak24_eval/per_image_metrics.jsonl`
- reconstructions: `results/stage4_cod_lite_adapter_eval/20260628_stage4_cod_lite_adapter_lpips020_alpha010_kodak24_eval/reconstructions`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_170219-t6wmruvt`
