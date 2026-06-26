# 20260627_stage1_semantic_vq_5k_pure_ae_l1_kodak_continuous_analysis

Date: 2026-06-27T03:17:09

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/analyze_stage1_semantic_vq.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_pure_ae_l1.pt --image-root /dpl/kodak --batch-size 4 --num-workers 4 --quantize-mix 0 --wandb-mode offline --run-name 20260627_stage1_semantic_vq_5k_pure_ae_l1_kodak_continuous_analysis
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_pure_ae_l1.pt
- quantize_mix: 0.0
- num_images: 24
- total_tokens: 1536
- active_codes_global: 361
- active_code_ratio_global: 0.0440673828125
- dead_code_ratio_global: 0.9559326171875
- global_code_entropy_bits: 6.234245777130127
- global_code_perplexity: 75.28266406017165
- mean_per_image_used_codes: 31.166666666666668
- p50_per_image_used_codes: 32.0
- max_per_image_used_codes: 42
- mean_soft_perplexity_batch: 51.22651672363281
- mean_psnr_sem: 21.333839178085327
- mean_l1_sem: 0.06236768700182438
- mean_ms_ssim_sem: 0.765045148630937
- mean_vq_mse: 3.4662486712137857
- latent_mean: 0.07368223244945209
- latent_std: 1.8815412322680156
- latent_min: -21.029640515645344
- latent_max: 23.239444414774578
- quantized_mean: 0.0031899401607612767
- quantized_std: 0.4937623490889867
- quantized_min: -1.9228780269622803
- quantized_max: 1.9984031915664673

## Artifacts

- analysis: `results/analysis/stage1_semantic_vq/20260627_stage1_semantic_vq_5k_pure_ae_l1_kodak_continuous_analysis/analysis.json`
- top_codes: `results/analysis/stage1_semantic_vq/20260627_stage1_semantic_vq_5k_pure_ae_l1_kodak_continuous_analysis/top_codes.csv`
- output_dir: `results/analysis/stage1_semantic_vq/20260627_stage1_semantic_vq_5k_pure_ae_l1_kodak_continuous_analysis`
