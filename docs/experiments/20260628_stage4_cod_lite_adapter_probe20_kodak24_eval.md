# 20260628_stage4_cod_lite_adapter_probe20_kodak24_eval

Date: 2026-06-28T16:23:32

## Command

```bash
scripts/eval_stage4_cod_lite_adapter.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_adapter_nativebase_probe20_kodak256.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_gencodec512_no_pp_recon_export_prefixsafe/reconstructions/manifest.jsonl --per-image-metrics results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_gencodec512_no_pp_recon_export_prefixsafe/per_image_metrics.jsonl --run-name 20260628_stage4_cod_lite_adapter_probe20_kodak24_eval --crop-size 512 --limit 24 --batch-size 1 --num-workers 2 --save-reconstructions --save-reconstruction-limit 8 --wandb-mode offline
```

## Summary

- actual_payload_bpp_mean: 0.014120737711588541
- stage3_psnr_mean: 21.667193094889324
- stage4_psnr_mean: 20.677699009577434
- stage3_ms_ssim_mean: 0.7225408007701238
- stage4_ms_ssim_mean: 0.6926118781169256
- stage3_lpips_alex_mean: 0.6366225394109885
- stage4_lpips_alex_mean: 0.5416211386521658
- stage3_dists_mean: 0.37316765636205673
- stage4_dists_mean: 0.3387455890576045
- stage4_l1_mean: 0.061723483726382256
- stage3_l1_mean: 0.05429107012848059
- count: 24
- stage4_psnr_win_rate: 0.0
- stage4_ms_ssim_win_rate: 0.0
- stage4_lpips_win_rate: 1.0
- stage4_dists_win_rate: 1.0
- stage4_payload_policy: inherits Stage 3 semantic/detail actual_payload_bpp; fixed CoD-Lite weights and adapter are not image-specific side information

## Artifacts

- summary: `results/stage4_cod_lite_adapter_eval/20260628_stage4_cod_lite_adapter_probe20_kodak24_eval/summary.json`
- per_image: `results/stage4_cod_lite_adapter_eval/20260628_stage4_cod_lite_adapter_probe20_kodak24_eval/per_image_metrics.jsonl`
- reconstructions: `results/stage4_cod_lite_adapter_eval/20260628_stage4_cod_lite_adapter_probe20_kodak24_eval/reconstructions`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_162325-jff0h1mq`
