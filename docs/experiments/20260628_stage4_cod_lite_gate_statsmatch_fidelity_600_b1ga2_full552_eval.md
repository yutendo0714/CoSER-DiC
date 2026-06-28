# 20260628_stage4_cod_lite_gate_statsmatch_fidelity_600_b1ga2_full552_eval

Date: 2026-06-28T22:29:59

## Command

```bash
scripts/eval_stage4_cod_lite_adapter.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_statsmatch_ft2k_b8.pt --gate-checkpoint checkpoints/stage4_cod_lite_gate/20260628_stage4_cod_lite_gate_statsmatch_fidelity_600_b1ga2.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl --per-image-metrics results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/per_image_metrics.jsonl --run-name 20260628_stage4_cod_lite_gate_statsmatch_fidelity_600_b1ga2_full552_eval --crop-size 512 --batch-size 1 --num-workers 2 --wandb-mode offline
```

## Summary

- actual_payload_bpp_mean: 0.013999109682829483
- stage3_psnr_mean: 21.99508856690448
- stage4_psnr_mean: 22.036746973576754
- stage3_ms_ssim_mean: 0.734825042899752
- stage4_ms_ssim_mean: 0.738374187975474
- stage3_lpips_alex_mean: 0.5757592567850066
- stage4_lpips_alex_mean: 0.5368684751143598
- stage3_dists_mean: 0.3536208531130915
- stage4_dists_mean: 0.3481400363903115
- stage4_l1_mean: 0.055097827554572425
- stage3_l1_mean: 0.05592005323969584
- condition_l1_mean: 0.41646764503009076
- base_condition_l1_mean: 0.5370908928630145
- condition_l1_delta_vs_base_mean: -0.12062324783292369
- condition_residual_l1_mean: 0.34370877393993776
- condition_delta_raw_l1_mean: 0.6618927205386369
- stage4_alpha_mean: 0.30705637171648553
- count: 552
- stage4_psnr_win_rate: 0.6068840579710145
- stage4_ms_ssim_win_rate: 0.7807971014492754
- stage4_lpips_win_rate: 0.9945652173913043
- stage4_dists_win_rate: 0.625
- stage4_blend_alpha: 1.0
- stage4_alpha_min: 0.1494140625
- stage4_alpha_max: 0.44140625
- stage4_alpha_std: 0.07726599276065826
- stage4_gate_checkpoint: checkpoints/stage4_cod_lite_gate/20260628_stage4_cod_lite_gate_statsmatch_fidelity_600_b1ga2.pt
- condition_residual_scale: 0.75
- condition_residual_tanh: True
- stage4_payload_policy: inherits Stage 3 semantic/detail actual_payload_bpp; fixed CoD-Lite weights, adapter, and deterministic decoder-side blend/gate alpha are not image-specific side information

## Artifacts

- summary: `results/stage4_cod_lite_adapter_eval/20260628_stage4_cod_lite_gate_statsmatch_fidelity_600_b1ga2_full552_eval/summary.json`
- per_image: `results/stage4_cod_lite_adapter_eval/20260628_stage4_cod_lite_gate_statsmatch_fidelity_600_b1ga2_full552_eval/per_image_metrics.jsonl`
- reconstructions: ``
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_222909-jiq40hve`
