# 20260627_stage1_semantic_vq_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only_kodak_analysis

Date: 2026-06-27T03:57:20

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/analyze_stage1_semantic_vq.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --image-root /dpl/kodak --batch-size 4 --num-workers 4 --quantize-mix 1 --wandb-mode offline --run-name 20260627_stage1_semantic_vq_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only_kodak_analysis
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- quantize_mix: 1.0
- num_images: 24
- total_tokens: 1536
- active_codes_global: 1064
- active_code_ratio_global: 0.1298828125
- dead_code_ratio_global: 0.8701171875
- global_code_entropy_bits: 9.83060073852539
- global_code_perplexity: 910.5539814592037
- mean_per_image_used_codes: 57.166666666666664
- p50_per_image_used_codes: 60.0
- max_per_image_used_codes: 62
- mean_soft_perplexity_batch: 227.77960205078125
- mean_psnr_sem: 20.736207803090412
- mean_l1_sem: 0.06508430729930599
- mean_ms_ssim_sem: 0.6757679680983225
- mean_vq_mse: 0.5266975313425064
- latent_mean: -0.006820335208127896
- latent_std: 1.9140191276868184
- latent_min: -22.89119307200114
- latent_max: 24.53454049428304
- quantized_mean: -0.011329780022303263
- quantized_std: 1.7965892155965169
- quantized_min: -21.814805030822754
- quantized_max: 21.818036715189617

## Artifacts

- analysis: `results/analysis/stage1_semantic_vq/20260627_stage1_semantic_vq_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only_kodak_analysis/analysis.json`
- top_codes: `results/analysis/stage1_semantic_vq/20260627_stage1_semantic_vq_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only_kodak_analysis/top_codes.csv`
- output_dir: `results/analysis/stage1_semantic_vq/20260627_stage1_semantic_vq_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only_kodak_analysis`
