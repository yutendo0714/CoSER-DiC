# 20260628_stage4_cod_lite_condition_gate_fidelity_probe300_b1ga2_full552_eval

Date: 2026-06-28T23:04:02

## Command

```bash
scripts/eval_stage4_cod_lite_adapter.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_statsmatch_ft2k_b8.pt --condition-gate-checkpoint checkpoints/stage4_cod_lite_condition_gate/20260628_stage4_cod_lite_condition_gate_fidelity_probe300_b1ga2.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl --per-image-metrics results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/per_image_metrics.jsonl --run-name 20260628_stage4_cod_lite_condition_gate_fidelity_probe300_b1ga2_full552_eval --crop-size 512 --batch-size 1 --num-workers 2 --wandb-mode offline
```

## Summary

- actual_payload_bpp_mean: 0.013999109682829483
- stage3_psnr_mean: 21.99508856690448
- stage4_psnr_mean: 21.316869879114456
- stage3_ms_ssim_mean: 0.734825042899752
- stage4_ms_ssim_mean: 0.7160698396788128
- stage3_lpips_alex_mean: 0.5757592567850066
- stage4_lpips_alex_mean: 0.5255005945775496
- stage3_dists_mean: 0.3536208531130915
- stage4_dists_mean: 0.3437182410903599
- stage4_l1_mean: 0.060011530409742525
- stage3_l1_mean: 0.05592005323969584
- condition_l1_mean: 0.46208032927867293
- ungated_condition_l1_mean: 0.41646764503009076
- base_condition_l1_mean: 0.5370908928630145
- condition_l1_delta_vs_base_mean: -0.07501056358434152
- condition_residual_l1_mean: 0.14740089148930882
- condition_delta_raw_l1_mean: 0.6618927205386369
- stage4_alpha_mean: 1.0
- condition_gate_mean_mean: 0.432723399521648
- count: 552
- stage4_psnr_win_rate: 0.012681159420289856
- stage4_ms_ssim_win_rate: 0.09963768115942029
- stage4_lpips_win_rate: 0.9764492753623188
- stage4_dists_win_rate: 0.6666666666666666
- stage4_blend_alpha: 1.0
- stage4_alpha_min: 1.0
- stage4_alpha_max: 1.0
- stage4_alpha_std: 0.0
- stage4_gate_checkpoint: 
- stage4_condition_gate_checkpoint: checkpoints/stage4_cod_lite_condition_gate/20260628_stage4_cod_lite_condition_gate_fidelity_probe300_b1ga2.pt
- condition_residual_scale: 0.75
- condition_residual_tanh: True
- stage4_payload_policy: inherits Stage 3 semantic/detail actual_payload_bpp; fixed CoD-Lite weights, adapter, deterministic decoder-side RGB diagnostic blend, or deterministic condition gate are not image-specific side information

## Artifacts

- summary: `results/stage4_cod_lite_adapter_eval/20260628_stage4_cod_lite_condition_gate_fidelity_probe300_b1ga2_full552_eval/summary.json`
- per_image: `results/stage4_cod_lite_adapter_eval/20260628_stage4_cod_lite_condition_gate_fidelity_probe300_b1ga2_full552_eval/per_image_metrics.jsonl`
- reconstructions: ``
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_230312-gjxcxo8n`
