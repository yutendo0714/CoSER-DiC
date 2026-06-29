# 20260628_stage4_mainline_ablation_smoke_limit2

Date: 2026-06-28T23:28:26

## Command

```bash
scripts/eval_stage4_cod_lite_adapter.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_statsmatch_ft2k_b8.pt --condition-gate-checkpoint checkpoints/stage4_cod_lite_condition_gate/20260628_stage4_cod_lite_condition_gate_fidelity_probe300_b1ga2.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl --per-image-metrics results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/per_image_metrics.jsonl --run-name 20260628_stage4_mainline_ablation_smoke_limit2 --crop-size 512 --limit 2 --batch-size 1 --num-workers 0 --semantic-latent-ablation zero --detail-context-ablation zero --condition-delta-ablation zero --wandb-mode offline
```

## Summary

- actual_payload_bpp_mean: 0.0135955810546875
- stage3_psnr_mean: 22.498653411865234
- stage4_psnr_mean: 22.085283279418945
- stage3_ms_ssim_mean: 0.6775427758693695
- stage4_ms_ssim_mean: 0.6525461077690125
- stage3_lpips_alex_mean: 0.7200557887554169
- stage4_lpips_alex_mean: 0.7259836196899414
- stage3_dists_mean: 0.4088239073753357
- stage4_dists_mean: 0.4140831530094147
- stage4_l1_mean: 0.05536314845085144
- stage3_l1_mean: 0.05272558331489563
- condition_l1_mean: 0.5622133612632751
- ungated_condition_l1_mean: 0.5622133612632751
- base_condition_l1_mean: 0.5622133612632751
- condition_l1_delta_vs_base_mean: 0.0
- condition_residual_l1_mean: 0.0
- condition_delta_raw_l1_mean: 0.0
- stage4_alpha_mean: 1.0
- condition_gate_mean_mean: 0.38152408599853516
- count: 2
- stage4_psnr_win_rate: 0.0
- stage4_ms_ssim_win_rate: 0.0
- stage4_lpips_win_rate: 0.5
- stage4_dists_win_rate: 0.5
- stage4_blend_alpha: 1.0
- stage4_alpha_min: 1.0
- stage4_alpha_max: 1.0
- stage4_alpha_std: 0.0
- stage4_gate_checkpoint: 
- stage4_condition_gate_checkpoint: checkpoints/stage4_cod_lite_condition_gate/20260628_stage4_cod_lite_condition_gate_fidelity_probe300_b1ga2.pt
- condition_residual_scale: 0.75
- condition_residual_tanh: True
- condition_delta_ablation: zero
- semantic_latent_ablation: zero
- detail_context_ablation: zero
- ablation_shuffle_seed: 1234
- stage4_payload_policy: inherits Stage 3 semantic/detail actual_payload_bpp; fixed CoD-Lite weights, adapter, deterministic decoder-side RGB diagnostic blend, or deterministic condition gate are not image-specific side information

## Artifacts

- summary: `results/stage4_cod_lite_adapter_eval/20260628_stage4_mainline_ablation_smoke_limit2/summary.json`
- per_image: `results/stage4_cod_lite_adapter_eval/20260628_stage4_mainline_ablation_smoke_limit2/per_image_metrics.jsonl`
- reconstructions: ``
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_232825-w8ht79t0`
