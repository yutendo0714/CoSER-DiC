# 20260628_stage4_cod_lite_adapter_eval_smoke2

Date: 2026-06-28T16:22:45

## Command

```bash
scripts/eval_stage4_cod_lite_adapter.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_adapter_nativebase_probe20_kodak256.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_gencodec512_no_pp_recon_export_prefixsafe/reconstructions/manifest.jsonl --per-image-metrics results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_gencodec512_no_pp_recon_export_prefixsafe/per_image_metrics.jsonl --run-name 20260628_stage4_cod_lite_adapter_eval_smoke2 --crop-size 512 --limit 2 --batch-size 1 --num-workers 0 --save-reconstructions --save-reconstruction-limit 2 --wandb-mode offline
```

## Summary

- actual_payload_bpp_mean: 0.0135955810546875
- stage3_psnr_mean: 22.498653411865234
- stage4_psnr_mean: 21.76042079925537
- stage3_ms_ssim_mean: 0.6775427758693695
- stage4_ms_ssim_mean: 0.6588469743728638
- stage3_lpips_alex_mean: 0.7200557887554169
- stage4_lpips_alex_mean: 0.6126459538936615
- stage3_dists_mean: 0.4088239073753357
- stage4_dists_mean: 0.3740125894546509
- stage4_l1_mean: 0.05680014193058014
- stage3_l1_mean: 0.05272558331489563
- count: 2
- stage4_psnr_win_rate: 0.0
- stage4_ms_ssim_win_rate: 0.0
- stage4_lpips_win_rate: 1.0
- stage4_dists_win_rate: 1.0
- stage4_payload_policy: inherits Stage 3 semantic/detail actual_payload_bpp; fixed CoD-Lite weights and adapter are not image-specific side information

## Artifacts

- summary: `results/stage4_cod_lite_adapter_eval/20260628_stage4_cod_lite_adapter_eval_smoke2/summary.json`
- per_image: `results/stage4_cod_lite_adapter_eval/20260628_stage4_cod_lite_adapter_eval_smoke2/per_image_metrics.jsonl`
- reconstructions: `results/stage4_cod_lite_adapter_eval/20260628_stage4_cod_lite_adapter_eval_smoke2/reconstructions`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_162243-4uq18gqs`
