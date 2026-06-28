# 20260628_stage4_cod_lite_gate_smoke2_eval4

Date: 2026-06-28T22:21:55

## Command

```bash
scripts/eval_stage4_cod_lite_adapter.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_statsmatch_ft2k_b8.pt --gate-checkpoint checkpoints/stage4_cod_lite_gate/20260628_stage4_cod_lite_gate_smoke2.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl --per-image-metrics results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/per_image_metrics.jsonl --run-name 20260628_stage4_cod_lite_gate_smoke2_eval4 --crop-size 512 --limit 4 --batch-size 1 --num-workers 0 --wandb-mode offline
```

## Summary

- actual_payload_bpp_mean: 0.01342010498046875
- stage3_psnr_mean: 23.542934894561768
- stage4_psnr_mean: 23.632684230804443
- stage3_ms_ssim_mean: 0.7483859658241272
- stage4_ms_ssim_mean: 0.7550816833972931
- stage3_lpips_alex_mean: 0.6308461055159569
- stage4_lpips_alex_mean: 0.5886473283171654
- stage3_dists_mean: 0.3717530518770218
- stage4_dists_mean: 0.3685555160045624
- stage4_l1_mean: 0.043241131119430065
- stage3_l1_mean: 0.044026040472090244
- condition_l1_mean: 0.42844579368829727
- base_condition_l1_mean: 0.539844736456871
- condition_l1_delta_vs_base_mean: -0.11139894276857376
- condition_residual_l1_mean: 0.34635206311941147
- condition_delta_raw_l1_mean: 0.6620350629091263
- stage4_alpha_mean: 0.30078125
- count: 4
- stage4_psnr_win_rate: 1.0
- stage4_ms_ssim_win_rate: 1.0
- stage4_lpips_win_rate: 1.0
- stage4_dists_win_rate: 0.25
- stage4_blend_alpha: 1.0
- stage4_alpha_min: 0.30078125
- stage4_alpha_max: 0.30078125
- stage4_alpha_std: 0.0
- stage4_gate_checkpoint: checkpoints/stage4_cod_lite_gate/20260628_stage4_cod_lite_gate_smoke2.pt
- condition_residual_scale: 0.75
- condition_residual_tanh: True
- stage4_payload_policy: inherits Stage 3 semantic/detail actual_payload_bpp; fixed CoD-Lite weights, adapter, and deterministic decoder-side blend/gate alpha are not image-specific side information

## Artifacts

- summary: `results/stage4_cod_lite_adapter_eval/20260628_stage4_cod_lite_gate_smoke2_eval4/summary.json`
- per_image: `results/stage4_cod_lite_adapter_eval/20260628_stage4_cod_lite_gate_smoke2_eval4/per_image_metrics.jsonl`
- reconstructions: ``
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_222153-5i8ses4t`
