# 20260628_stage4_detailaware_adapter_bpp0312_transfer_ft600_b2_limit64_eval

Date: 2026-06-29T00:19:12

## Command

```bash
scripts/eval_stage4_cod_lite_adapter.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_detailaware_adapter_bpp0312_transfer_ft600_b2.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl --per-image-metrics results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/per_image_metrics.jsonl --run-name 20260628_stage4_detailaware_adapter_bpp0312_transfer_ft600_b2_limit64_eval --crop-size 512 --limit 64 --batch-size 2 --num-workers 2 --blend-alpha 1.0 --wandb-mode offline
```

## Summary

- actual_payload_bpp_mean: 0.013654232025146484
- stage3_psnr_mean: 22.36516010761261
- stage4_psnr_mean: 21.83790247142315
- stage3_ms_ssim_mean: 0.7568008005619049
- stage4_ms_ssim_mean: 0.7441892446950078
- stage3_lpips_alex_mean: 0.5449693585978821
- stage4_lpips_alex_mean: 0.4214613934746012
- stage3_dists_mean: 0.3497471623122692
- stage4_dists_mean: 0.2888367474079132
- stage4_l1_mean: 0.05373372651229147
- stage3_l1_mean: 0.05090900382492691
- condition_l1_mean: 0.45420486805960536
- ungated_condition_l1_mean: 0.45420486805960536
- base_condition_l1_mean: 0.5807610866613686
- condition_l1_delta_vs_base_mean: -0.12655621860176325
- condition_residual_l1_mean: 0.3723017261363566
- condition_delta_raw_l1_mean: 0.592245880048722
- stage4_alpha_mean: 1.0
- condition_gate_mean_mean: 0.0
- count: 64
- stage4_psnr_win_rate: 0.015625
- stage4_ms_ssim_win_rate: 0.15625
- stage4_lpips_win_rate: 0.96875
- stage4_dists_win_rate: 0.984375
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
- detail_context_ablation: normal
- ablation_shuffle_seed: 1234
- stage4_payload_policy: inherits Stage 3 semantic/detail actual_payload_bpp; fixed CoD-Lite weights, adapter, deterministic decoder-side RGB diagnostic blend, or deterministic condition gate are not image-specific side information

## Artifacts

- summary: `results/stage4_cod_lite_adapter_eval/20260628_stage4_detailaware_adapter_bpp0312_transfer_ft600_b2_limit64_eval/summary.json`
- per_image: `results/stage4_cod_lite_adapter_eval/20260628_stage4_detailaware_adapter_bpp0312_transfer_ft600_b2_limit64_eval/per_image_metrics.jsonl`
- reconstructions: ``
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_001905-l4c13hbb`
