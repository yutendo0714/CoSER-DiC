# 20260628_stage4_cod_lite_pyramid_sem256_detailctx6_tanh075_traincache2048_5k_b8_full552_eval_save552

Date: 2026-06-28T21:06:31

## Command

```bash
scripts/eval_stage4_cod_lite_adapter.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_tanh075_traincache2048_5k_b8.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl --per-image-metrics results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/per_image_metrics.jsonl --run-name 20260628_stage4_cod_lite_pyramid_sem256_detailctx6_tanh075_traincache2048_5k_b8_full552_eval_save552 --crop-size 512 --batch-size 1 --num-workers 2 --blend-alpha 1.0 --save-reconstructions --wandb-mode offline
```

## Summary

- actual_payload_bpp_mean: 0.013999109682829483
- stage3_psnr_mean: 21.99508856690448
- stage4_psnr_mean: 21.166184126466945
- stage3_ms_ssim_mean: 0.734825042899752
- stage4_ms_ssim_mean: 0.7064123481943987
- stage3_lpips_alex_mean: 0.5757592567850066
- stage4_lpips_alex_mean: 0.4487357784562029
- stage3_dists_mean: 0.3536208531130915
- stage4_dists_mean: 0.31286428069722827
- stage4_l1_mean: 0.060640172692069755
- stage3_l1_mean: 0.05592005323969584
- condition_l1_mean: 0.41116778849475627
- base_condition_l1_mean: 0.5370908928630145
- condition_l1_delta_vs_base_mean: -0.1259231043682582
- condition_residual_l1_mean: 0.32379427193191607
- condition_delta_raw_l1_mean: 0.5886345268159673
- count: 552
- stage4_psnr_win_rate: 0.009057971014492754
- stage4_ms_ssim_win_rate: 0.050724637681159424
- stage4_lpips_win_rate: 0.9927536231884058
- stage4_dists_win_rate: 0.8858695652173914
- stage4_blend_alpha: 1.0
- condition_residual_scale: 0.75
- condition_residual_tanh: True
- stage4_payload_policy: inherits Stage 3 semantic/detail actual_payload_bpp; fixed CoD-Lite weights, adapter, and deterministic decoder-side blend alpha are not image-specific side information

## Artifacts

- summary: `results/stage4_cod_lite_adapter_eval/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_tanh075_traincache2048_5k_b8_full552_eval_save552/summary.json`
- per_image: `results/stage4_cod_lite_adapter_eval/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_tanh075_traincache2048_5k_b8_full552_eval_save552/per_image_metrics.jsonl`
- reconstructions: `results/stage4_cod_lite_adapter_eval/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_tanh075_traincache2048_5k_b8_full552_eval_save552/reconstructions`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_210050-89csvtd4`
