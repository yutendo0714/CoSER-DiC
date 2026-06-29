# 20260628_stage4_detailaware_adapter_bpp0312_transfer_ft600_b2_full552_save_eval

Date: 2026-06-29T00:23:58

## Command

```bash
scripts/eval_stage4_cod_lite_adapter.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_detailaware_adapter_bpp0312_transfer_ft600_b2.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl --per-image-metrics results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/per_image_metrics.jsonl --run-name 20260628_stage4_detailaware_adapter_bpp0312_transfer_ft600_b2_full552_save_eval --crop-size 512 --batch-size 2 --num-workers 2 --blend-alpha 1.0 --save-reconstructions --save-reconstruction-kinds reference,stage4 --wandb-mode offline
```

## Summary

- actual_payload_bpp_mean: 0.013999109682829483
- stage3_psnr_mean: 21.99508856690448
- stage4_psnr_mean: 21.44379132726918
- stage3_ms_ssim_mean: 0.7348250426298034
- stage4_ms_ssim_mean: 0.7216467787174211
- stage3_lpips_alex_mean: 0.5757592567850066
- stage4_lpips_alex_mean: 0.4402133921107304
- stage3_dists_mean: 0.3536208531130915
- stage4_dists_mean: 0.29167848728273227
- stage4_l1_mean: 0.05876064179735123
- stage3_l1_mean: 0.05592005323969584
- condition_l1_mean: 0.45888399824068166
- ungated_condition_l1_mean: 0.45888399824068166
- base_condition_l1_mean: 0.5991991752947586
- condition_l1_delta_vs_base_mean: -0.14031517705407695
- condition_residual_l1_mean: 0.37631069544864737
- condition_delta_raw_l1_mean: 0.6032781374195347
- stage4_alpha_mean: 1.0
- condition_gate_mean_mean: 0.0
- count: 552
- stage4_psnr_win_rate: 0.016304347826086956
- stage4_ms_ssim_win_rate: 0.1213768115942029
- stage4_lpips_win_rate: 0.9782608695652174
- stage4_dists_win_rate: 0.9963768115942029
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
- save_reconstruction_kinds: ['reference', 'stage4']
- stage4_payload_policy: inherits Stage 3 semantic/detail actual_payload_bpp; fixed CoD-Lite weights, adapter, deterministic decoder-side RGB diagnostic blend, or deterministic condition gate are not image-specific side information

## Artifacts

- summary: `results/stage4_cod_lite_adapter_eval/20260628_stage4_detailaware_adapter_bpp0312_transfer_ft600_b2_full552_save_eval/summary.json`
- per_image: `results/stage4_cod_lite_adapter_eval/20260628_stage4_detailaware_adapter_bpp0312_transfer_ft600_b2_full552_save_eval/per_image_metrics.jsonl`
- reconstructions: `results/stage4_cod_lite_adapter_eval/20260628_stage4_detailaware_adapter_bpp0312_transfer_ft600_b2_full552_save_eval/reconstructions`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_002151-ron3d6kf`
