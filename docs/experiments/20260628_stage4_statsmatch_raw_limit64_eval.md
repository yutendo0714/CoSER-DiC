# 20260628_stage4_statsmatch_raw_limit64_eval

Date: 2026-06-28T23:36:19

## Command

```bash
scripts/eval_stage4_cod_lite_adapter.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_statsmatch_ft2k_b8.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl --per-image-metrics results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/per_image_metrics.jsonl --run-name 20260628_stage4_statsmatch_raw_limit64_eval --crop-size 512 --limit 64 --batch-size 1 --num-workers 2 --blend-alpha 1.0 --wandb-mode offline
```

## Summary

- actual_payload_bpp_mean: 0.013654232025146484
- stage3_psnr_mean: 22.36516010761261
- stage4_psnr_mean: 21.510314717888832
- stage3_ms_ssim_mean: 0.7568008033558726
- stage4_ms_ssim_mean: 0.7363501256331801
- stage3_lpips_alex_mean: 0.5449693585978821
- stage4_lpips_alex_mean: 0.4212654841830954
- stage3_dists_mean: 0.3497471623122692
- stage4_dists_mean: 0.3018353972584009
- stage4_l1_mean: 0.05591533786355285
- stage3_l1_mean: 0.05090900382492691
- condition_l1_mean: 0.41765901632606983
- ungated_condition_l1_mean: 0.41765901632606983
- base_condition_l1_mean: 0.5285057057626545
- condition_l1_delta_vs_base_mean: -0.11084668943658471
- condition_residual_l1_mean: 0.34145683236420155
- condition_delta_raw_l1_mean: 0.6505557466298342
- stage4_alpha_mean: 1.0
- condition_gate_mean_mean: 0.0
- count: 64
- stage4_psnr_win_rate: 0.015625
- stage4_ms_ssim_win_rate: 0.109375
- stage4_lpips_win_rate: 0.984375
- stage4_dists_win_rate: 0.9375
- stage4_blend_alpha: 1.0
- stage4_alpha_min: 1.0
- stage4_alpha_max: 1.0
- stage4_alpha_std: 0.0
- stage4_gate_checkpoint: 
- stage4_condition_gate_checkpoint: 
- condition_residual_scale: 0.75
- condition_residual_tanh: True
- condition_delta_ablation: normal
- semantic_latent_ablation: normal
- detail_context_ablation: normal
- ablation_shuffle_seed: 1234
- stage4_payload_policy: inherits Stage 3 semantic/detail actual_payload_bpp; fixed CoD-Lite weights, adapter, deterministic decoder-side RGB diagnostic blend, or deterministic condition gate are not image-specific side information

## Artifacts

- summary: `results/stage4_cod_lite_adapter_eval/20260628_stage4_statsmatch_raw_limit64_eval/summary.json`
- per_image: `results/stage4_cod_lite_adapter_eval/20260628_stage4_statsmatch_raw_limit64_eval/per_image_metrics.jsonl`
- reconstructions: ``
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_233613-retrb9g0`
