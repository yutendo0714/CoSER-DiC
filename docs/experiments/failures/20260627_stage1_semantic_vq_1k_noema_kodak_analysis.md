# 20260627_stage1_semantic_vq_1k_noema_kodak_analysis

Date: 2026-06-27T02:50:03

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/analyze_stage1_semantic_vq.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_1k_noema.pt --image-root /dpl/kodak --max-images 24 --batch-size 4 --num-workers 4 --wandb-mode offline --run-name 20260627_stage1_semantic_vq_1k_noema_kodak_analysis
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_1k_noema.pt
- num_images: 24
- total_tokens: 1536
- active_codes_global: 36
- active_code_ratio_global: 0.00439453125
- dead_code_ratio_global: 0.99560546875
- global_code_entropy_bits: 4.916082859039307
- global_code_perplexity: 30.191757997476234
- mean_per_image_used_codes: 35.833333333333336
- p50_per_image_used_codes: 36.0
- max_per_image_used_codes: 36
- mean_soft_perplexity_batch: 503.29686482747394
- mean_psnr_sem: 4.498203635215759
- mean_l1_sem: 0.5668156643708547
- mean_ms_ssim_sem: 0.2785754098246495
- mean_vq_mse: 0.14025204877058664
- latent_mean: -0.002937911117138962
- latent_std: 0.0697111946841081
- latent_min: -0.3064857174952825
- latent_max: 0.34487539529800415
- quantized_mean: 0.0005106745811644942
- quantized_std: 0.3794913937648137
- quantized_min: -1.62942636013031
- quantized_max: 1.8330032030741374

## Artifacts

- analysis: `results/analysis/stage1_semantic_vq/20260627_stage1_semantic_vq_1k_noema_kodak_analysis/analysis.json`
- top_codes: `results/analysis/stage1_semantic_vq/20260627_stage1_semantic_vq_1k_noema_kodak_analysis/top_codes.csv`
- output_dir: `results/analysis/stage1_semantic_vq/20260627_stage1_semantic_vq_1k_noema_kodak_analysis`
