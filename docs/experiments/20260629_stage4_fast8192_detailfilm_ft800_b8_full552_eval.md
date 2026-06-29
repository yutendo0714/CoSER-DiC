# 20260629_stage4_fast8192_detailfilm_ft800_b8_full552_eval

Date: 2026-06-29T03:19:01

## Command

```bash
scripts/eval_stage4_cod_lite_adapter.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailfilm_ft800_b8.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl --per-image-metrics results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/per_image_metrics.jsonl --run-name 20260629_stage4_fast8192_detailfilm_ft800_b8_full552_eval --crop-size 512 --batch-size 4 --num-workers 4 --blend-alpha 1.0 --wandb-mode offline
```

## Summary

- actual_payload_bpp_mean: 0.013999109682829483
- stage3_psnr_mean: 21.995088589364205
- stage4_psnr_mean: 21.24270937408226
- stage3_ms_ssim_mean: 0.7348250424138446
- stage4_ms_ssim_mean: 0.7150113355854283
- stage3_lpips_alex_mean: 0.5757592567850066
- stage4_lpips_alex_mean: 0.430416506546401
- stage3_dists_mean: 0.3536208531130915
- stage4_dists_mean: 0.2982114157167034
- stage4_l1_mean: 0.059719513656328556
- stage3_l1_mean: 0.05592005335442398
- condition_l1_mean: 0.40597158119730326
- ungated_condition_l1_mean: 0.40597158119730326
- base_condition_l1_mean: 0.53720763368883
- condition_l1_delta_vs_base_mean: -0.13123605249152667
- condition_residual_l1_mean: 0.3338726160750873
- condition_delta_raw_l1_mean: 0.5207213420366895
- stage4_alpha_mean: 1.0
- condition_gate_mean_mean: 0.0
- count: 552
- stage4_psnr_win_rate: 0.014492753623188406
- stage4_ms_ssim_win_rate: 0.10326086956521739
- stage4_lpips_win_rate: 0.9945652173913043
- stage4_dists_win_rate: 0.9655797101449275
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
- save_reconstruction_kinds: ['reference', 'semantic_only', 'stage3', 'stage4', 'quad']
- stage4_payload_policy: inherits Stage 3 semantic/detail actual_payload_bpp; fixed CoD-Lite weights, adapter, deterministic decoder-side RGB diagnostic blend, or deterministic condition gate are not image-specific side information

## Artifacts

- summary: `results/stage4_cod_lite_adapter_eval/20260629_stage4_fast8192_detailfilm_ft800_b8_full552_eval/summary.json`
- per_image: `results/stage4_cod_lite_adapter_eval/20260629_stage4_fast8192_detailfilm_ft800_b8_full552_eval/per_image_metrics.jsonl`
- reconstructions: ``
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_031817-f6xsof7b`
