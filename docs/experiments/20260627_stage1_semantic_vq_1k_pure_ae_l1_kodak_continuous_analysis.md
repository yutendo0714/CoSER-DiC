# 20260627_stage1_semantic_vq_1k_pure_ae_l1_kodak_continuous_analysis

Date: 2026-06-27T03:06:03

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/analyze_stage1_semantic_vq.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_1k_pure_ae_l1_probe.pt --image-root /dpl/kodak --max-images 24 --batch-size 8 --num-workers 4 --quantize-mix 0 --wandb-mode offline --run-name 20260627_stage1_semantic_vq_1k_pure_ae_l1_kodak_continuous_analysis
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_1k_pure_ae_l1_probe.pt
- quantize_mix: 0.0
- num_images: 24
- total_tokens: 1536
- active_codes_global: 260
- active_code_ratio_global: 0.03173828125
- dead_code_ratio_global: 0.96826171875
- global_code_entropy_bits: 5.835705757141113
- global_code_perplexity: 57.11135691905353
- mean_per_image_used_codes: 28.583333333333332
- p50_per_image_used_codes: 29.0
- max_per_image_used_codes: 45
- mean_soft_perplexity_batch: 52.833082834879555
- mean_psnr_sem: 19.76530929406484
- mean_l1_sem: 0.07688564745088418
- mean_ms_ssim_sem: 0.6262773498892784
- mean_vq_mse: 2.749620278676351
- latent_mean: 0.04076233630379041
- latent_std: 1.6875187555948894
- latent_min: -14.347432454427084
- latent_max: 16.104432106018066
- quantized_mean: 0.007710364491989215
- quantized_std: 0.4850987096627553
- quantized_min: -1.8893696467081706
- quantized_max: 2.031080881754557

## Artifacts

- analysis: `results/analysis/stage1_semantic_vq/20260627_stage1_semantic_vq_1k_pure_ae_l1_kodak_continuous_analysis/analysis.json`
- top_codes: `results/analysis/stage1_semantic_vq/20260627_stage1_semantic_vq_1k_pure_ae_l1_kodak_continuous_analysis/top_codes.csv`
- output_dir: `results/analysis/stage1_semantic_vq/20260627_stage1_semantic_vq_1k_pure_ae_l1_kodak_continuous_analysis`
