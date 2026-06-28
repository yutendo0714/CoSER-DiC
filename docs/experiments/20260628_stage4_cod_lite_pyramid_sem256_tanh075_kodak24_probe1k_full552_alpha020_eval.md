# 20260628_stage4_cod_lite_pyramid_sem256_tanh075_kodak24_probe1k_full552_alpha020_eval

Date: 2026-06-28T18:37:36

## Command

```bash
scripts/eval_stage4_cod_lite_adapter.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_tanh075_kodak24_probe1k.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl --per-image-metrics results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/per_image_metrics.jsonl --run-name 20260628_stage4_cod_lite_pyramid_sem256_tanh075_kodak24_probe1k_full552_alpha020_eval --crop-size 512 --batch-size 1 --num-workers 2 --blend-alpha 0.2 --wandb-mode offline
```

## Summary

- actual_payload_bpp_mean: 0.013999109682829483
- stage3_psnr_mean: 21.99508856690448
- stage4_psnr_mean: 21.99580141081326
- stage3_ms_ssim_mean: 0.734825042899752
- stage4_ms_ssim_mean: 0.7368021408986786
- stage3_lpips_alex_mean: 0.5757592567850066
- stage4_lpips_alex_mean: 0.5559751651236329
- stage3_dists_mean: 0.3536208531130915
- stage4_dists_mean: 0.3513041531694108
- stage4_l1_mean: 0.05577512849223516
- stage3_l1_mean: 0.05592005323969584
- condition_l1_mean: 0.47107743963167287
- base_condition_l1_mean: 0.5370908928630145
- condition_l1_delta_vs_base_mean: -0.0660134532313416
- condition_residual_l1_mean: 0.3612815095786599
- condition_delta_raw_l1_mean: 0.7299986536736074
- count: 552
- stage4_psnr_win_rate: 0.4438405797101449
- stage4_ms_ssim_win_rate: 0.8260869565217391
- stage4_lpips_win_rate: 0.9963768115942029
- stage4_dists_win_rate: 0.6322463768115942
- stage4_blend_alpha: 0.2
- condition_residual_scale: 0.75
- condition_residual_tanh: True
- stage4_payload_policy: inherits Stage 3 semantic/detail actual_payload_bpp; fixed CoD-Lite weights, adapter, and deterministic decoder-side blend alpha are not image-specific side information

## Artifacts

- summary: `results/stage4_cod_lite_adapter_eval/20260628_stage4_cod_lite_pyramid_sem256_tanh075_kodak24_probe1k_full552_alpha020_eval/summary.json`
- per_image: `results/stage4_cod_lite_adapter_eval/20260628_stage4_cod_lite_pyramid_sem256_tanh075_kodak24_probe1k_full552_alpha020_eval/per_image_metrics.jsonl`
- reconstructions: ``
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_183646-2u479slt`
