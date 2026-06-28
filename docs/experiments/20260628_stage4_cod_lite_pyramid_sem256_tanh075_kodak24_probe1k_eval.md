# 20260628_stage4_cod_lite_pyramid_sem256_tanh075_kodak24_probe1k_eval

Date: 2026-06-28T17:58:35

## Command

```bash
scripts/eval_stage4_cod_lite_adapter.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_tanh075_kodak24_probe1k.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_kodak24_feature_cache_smoke/reconstructions/manifest.jsonl --per-image-metrics results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_kodak24_feature_cache_smoke/per_image_metrics.jsonl --run-name 20260628_stage4_cod_lite_pyramid_sem256_tanh075_kodak24_probe1k_eval --limit 24 --crop-size 512 --batch-size 1 --num-workers 2 --blend-alpha 1.0 --save-reconstructions --save-reconstruction-limit 24 --wandb-mode offline
```

## Summary

- actual_payload_bpp_mean: 0.014123280843098959
- stage3_psnr_mean: 21.6673747698466
- stage4_psnr_mean: 20.645820418993633
- stage3_ms_ssim_mean: 0.7225482811530431
- stage4_ms_ssim_mean: 0.6911284402012825
- stage3_lpips_alex_mean: 0.6366176729400953
- stage4_lpips_alex_mean: 0.4294225325187047
- stage3_dists_mean: 0.3731629326939583
- stage4_dists_mean: 0.2743087311585744
- stage4_l1_mean: 0.06159543013200164
- stage3_l1_mean: 0.05428991994510094
- count: 24
- stage4_psnr_win_rate: 0.0
- stage4_ms_ssim_win_rate: 0.0
- stage4_lpips_win_rate: 1.0
- stage4_dists_win_rate: 1.0
- stage4_blend_alpha: 1.0
- condition_residual_scale: 0.75
- condition_residual_tanh: True
- stage4_payload_policy: inherits Stage 3 semantic/detail actual_payload_bpp; fixed CoD-Lite weights, adapter, and deterministic decoder-side blend alpha are not image-specific side information

## Artifacts

- summary: `results/stage4_cod_lite_adapter_eval/20260628_stage4_cod_lite_pyramid_sem256_tanh075_kodak24_probe1k_eval/summary.json`
- per_image: `results/stage4_cod_lite_adapter_eval/20260628_stage4_cod_lite_pyramid_sem256_tanh075_kodak24_probe1k_eval/per_image_metrics.jsonl`
- reconstructions: `results/stage4_cod_lite_adapter_eval/20260628_stage4_cod_lite_pyramid_sem256_tanh075_kodak24_probe1k_eval/reconstructions`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_175819-t95c1lqa`
