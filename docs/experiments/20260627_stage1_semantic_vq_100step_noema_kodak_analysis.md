# 20260627_stage1_semantic_vq_100step_noema_kodak_analysis

Date: 2026-06-27T02:48:34

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/analyze_stage1_semantic_vq.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_100step_noema.pt --image-root /dpl/kodak --max-images 24 --batch-size 4 --num-workers 4 --wandb-mode offline --run-name 20260627_stage1_semantic_vq_100step_noema_kodak_analysis
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_100step_noema.pt
- num_images: 24
- total_tokens: 1536
- active_codes_global: 99
- active_code_ratio_global: 0.0120849609375
- dead_code_ratio_global: 0.9879150390625
- global_code_entropy_bits: 4.301666736602783
- global_code_perplexity: 19.721081099954976
- mean_per_image_used_codes: 21.458333333333332
- p50_per_image_used_codes: 21.0
- max_per_image_used_codes: 30
- mean_soft_perplexity_batch: 23.89229615529378
- mean_psnr_sem: 13.100402573744455
- mean_l1_sem: 0.19186307241519293
- mean_ms_ssim_sem: 0.30433466502775747
- mean_vq_mse: 0.3736657202243805
- latent_mean: 0.02136974036693573
- latent_std: 0.5480422874291738
- latent_min: -3.1408467292785645
- latent_max: 3.7709653774897256
- quantized_mean: 0.008708583501478037
- quantized_std: 0.4723732074101766
- quantized_min: -1.8705228567123413
- quantized_max: 1.92730712890625

## Artifacts

- analysis: `results/analysis/stage1_semantic_vq/20260627_stage1_semantic_vq_100step_noema_kodak_analysis/analysis.json`
- top_codes: `results/analysis/stage1_semantic_vq/20260627_stage1_semantic_vq_100step_noema_kodak_analysis/top_codes.csv`
- output_dir: `results/analysis/stage1_semantic_vq/20260627_stage1_semantic_vq_100step_noema_kodak_analysis`
