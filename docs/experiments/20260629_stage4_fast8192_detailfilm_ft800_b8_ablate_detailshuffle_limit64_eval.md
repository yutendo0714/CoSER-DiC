# 20260629_stage4_fast8192_detailfilm_ft800_b8_ablate_detailshuffle_limit64_eval

Date: 2026-06-29T03:17:54

## Command

```bash
scripts/eval_stage4_cod_lite_adapter.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailfilm_ft800_b8.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl --per-image-metrics results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/per_image_metrics.jsonl --run-name 20260629_stage4_fast8192_detailfilm_ft800_b8_ablate_detailshuffle_limit64_eval --crop-size 512 --limit 64 --batch-size 4 --num-workers 4 --blend-alpha 1.0 --detail-context-ablation shuffle --wandb-mode offline
```

## Summary

- actual_payload_bpp_mean: 0.013654232025146484
- stage3_psnr_mean: 22.365160256624222
- stage4_psnr_mean: 21.652474999427795
- stage3_ms_ssim_mean: 0.7568007968366146
- stage4_ms_ssim_mean: 0.7385989362373948
- stage3_lpips_alex_mean: 0.5449693585978821
- stage4_lpips_alex_mean: 0.4108318519720342
- stage3_dists_mean: 0.3497471623122692
- stage4_dists_mean: 0.2981176823377609
- stage4_l1_mean: 0.054896396410185844
- stage3_l1_mean: 0.05090900359209627
- condition_l1_mean: 0.4066776339896023
- ungated_condition_l1_mean: 0.4066776339896023
- base_condition_l1_mean: 0.5287581100128591
- condition_l1_delta_vs_base_mean: -0.12208047602325678
- condition_residual_l1_mean: 0.32750219758599997
- condition_delta_raw_l1_mean: 0.5023415302857757
- stage4_alpha_mean: 1.0
- condition_gate_mean_mean: 0.0
- count: 64
- stage4_psnr_win_rate: 0.015625
- stage4_ms_ssim_win_rate: 0.125
- stage4_lpips_win_rate: 0.984375
- stage4_dists_win_rate: 0.953125
- stage4_blend_alpha: 1.0
- stage4_alpha_min: 1.0
- stage4_alpha_max: 1.0
- stage4_alpha_std: 0.0
- stage4_gate_checkpoint: 
- stage4_condition_gate_checkpoint: 
- condition_residual_scale: 0.85
- condition_residual_tanh: True
- condition_delta_ablation: normal
- semantic_latent_ablation: normal
- detail_context_ablation: shuffle
- ablation_shuffle_seed: 1234
- save_reconstruction_kinds: ['reference', 'semantic_only', 'stage3', 'stage4', 'quad']
- stage4_payload_policy: inherits Stage 3 semantic/detail actual_payload_bpp; fixed CoD-Lite weights, adapter, deterministic decoder-side RGB diagnostic blend, or deterministic condition gate are not image-specific side information

## Artifacts

- summary: `results/stage4_cod_lite_adapter_eval/20260629_stage4_fast8192_detailfilm_ft800_b8_ablate_detailshuffle_limit64_eval/summary.json`
- per_image: `results/stage4_cod_lite_adapter_eval/20260629_stage4_fast8192_detailfilm_ft800_b8_ablate_detailshuffle_limit64_eval/per_image_metrics.jsonl`
- reconstructions: ``
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_031744-okebe6vk`
