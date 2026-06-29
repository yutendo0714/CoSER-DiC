# 20260629_stage4_fast8192_detailcontrol_semdrop020_ft800_b8_ablate_detailzero_full552_eval

Date: 2026-06-29T02:23:21

## Command

```bash
scripts/eval_stage4_cod_lite_adapter.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailcontrol_semdrop020_ft800_b8.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl --per-image-metrics results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/per_image_metrics.jsonl --run-name 20260629_stage4_fast8192_detailcontrol_semdrop020_ft800_b8_ablate_detailzero_full552_eval --crop-size 512 --batch-size 2 --num-workers 2 --blend-alpha 1.0 --detail-context-ablation zero --wandb-mode offline
```

## Summary

- actual_payload_bpp_mean: 0.013999109682829483
- stage3_psnr_mean: 21.99508856690448
- stage4_psnr_mean: 21.25387437965559
- stage3_ms_ssim_mean: 0.7348250426298034
- stage4_ms_ssim_mean: 0.7151620185434602
- stage3_lpips_alex_mean: 0.5757592567850066
- stage4_lpips_alex_mean: 0.43611408322644624
- stage3_dists_mean: 0.3536208531130915
- stage4_dists_mean: 0.30227842171123065
- stage4_l1_mean: 0.05966626256670587
- stage3_l1_mean: 0.05592005323969584
- condition_l1_mean: 0.4078974273001802
- ungated_condition_l1_mean: 0.4078974273001802
- base_condition_l1_mean: 0.537207702309757
- condition_l1_delta_vs_base_mean: -0.1293102750095768
- condition_residual_l1_mean: 0.32708297783265944
- condition_delta_raw_l1_mean: 0.501496316784102
- stage4_alpha_mean: 1.0
- condition_gate_mean_mean: 0.0
- count: 552
- stage4_psnr_win_rate: 0.014492753623188406
- stage4_ms_ssim_win_rate: 0.10869565217391304
- stage4_lpips_win_rate: 0.9963768115942029
- stage4_dists_win_rate: 0.9619565217391305
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
- detail_context_ablation: zero
- ablation_shuffle_seed: 1234
- save_reconstruction_kinds: ['reference', 'semantic_only', 'stage3', 'stage4', 'quad']
- stage4_payload_policy: inherits Stage 3 semantic/detail actual_payload_bpp; fixed CoD-Lite weights, adapter, deterministic decoder-side RGB diagnostic blend, or deterministic condition gate are not image-specific side information

## Artifacts

- summary: `results/stage4_cod_lite_adapter_eval/20260629_stage4_fast8192_detailcontrol_semdrop020_ft800_b8_ablate_detailzero_full552_eval/summary.json`
- per_image: `results/stage4_cod_lite_adapter_eval/20260629_stage4_fast8192_detailcontrol_semdrop020_ft800_b8_ablate_detailzero_full552_eval/per_image_metrics.jsonl`
- reconstructions: ``
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_022201-qnqzk8jo`
