# 20260628_stage4_statsmatch_ft2k_full552_raw_save_eval

Date: 2026-06-28T22:07:48

## Command

```bash
scripts/eval_stage4_cod_lite_adapter.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_statsmatch_ft2k_b8.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl --per-image-metrics results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/per_image_metrics.jsonl --run-name 20260628_stage4_statsmatch_ft2k_full552_raw_save_eval --crop-size 512 --batch-size 1 --num-workers 2 --blend-alpha 1.0 --save-reconstructions --wandb-mode offline
```

## Summary

- actual_payload_bpp_mean: 0.013999109682829483
- stage3_psnr_mean: 21.99508856690448
- stage4_psnr_mean: 21.115692321804985
- stage3_ms_ssim_mean: 0.734825042899752
- stage4_ms_ssim_mean: 0.7125597291666529
- stage3_lpips_alex_mean: 0.5757592567850066
- stage4_lpips_alex_mean: 0.4427773645552604
- stage3_dists_mean: 0.3536208531130915
- stage4_dists_mean: 0.3040213321430096
- stage4_l1_mean: 0.06059508968188522
- stage3_l1_mean: 0.05592005323969584
- condition_l1_mean: 0.41646764503009076
- base_condition_l1_mean: 0.5370908928630145
- condition_l1_delta_vs_base_mean: -0.12062324783292369
- condition_residual_l1_mean: 0.34370877393993776
- condition_delta_raw_l1_mean: 0.6618927205386369
- count: 552
- stage4_psnr_win_rate: 0.014492753623188406
- stage4_ms_ssim_win_rate: 0.08695652173913043
- stage4_lpips_win_rate: 0.9909420289855072
- stage4_dists_win_rate: 0.9492753623188406
- stage4_blend_alpha: 1.0
- condition_residual_scale: 0.75
- condition_residual_tanh: True
- stage4_payload_policy: inherits Stage 3 semantic/detail actual_payload_bpp; fixed CoD-Lite weights, adapter, and deterministic decoder-side blend alpha are not image-specific side information

## Artifacts

- summary: `results/stage4_cod_lite_adapter_eval/20260628_stage4_statsmatch_ft2k_full552_raw_save_eval/summary.json`
- per_image: `results/stage4_cod_lite_adapter_eval/20260628_stage4_statsmatch_ft2k_full552_raw_save_eval/per_image_metrics.jsonl`
- reconstructions: `results/stage4_cod_lite_adapter_eval/20260628_stage4_statsmatch_ft2k_full552_raw_save_eval/reconstructions`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_220207-nlxqymwy`
