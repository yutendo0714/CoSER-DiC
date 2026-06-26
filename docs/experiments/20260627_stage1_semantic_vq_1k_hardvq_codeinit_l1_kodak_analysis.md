# 20260627_stage1_semantic_vq_1k_hardvq_codeinit_l1_kodak_analysis

Date: 2026-06-27T03:08:33

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/analyze_stage1_semantic_vq.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_1k_hardvq_from_ae_codeinit_l1.pt --image-root /dpl/kodak --max-images 24 --batch-size 8 --num-workers 4 --quantize-mix 1 --wandb-mode offline --run-name 20260627_stage1_semantic_vq_1k_hardvq_codeinit_l1_kodak_analysis
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_1k_hardvq_from_ae_codeinit_l1.pt
- quantize_mix: 1.0
- num_images: 24
- total_tokens: 1536
- active_codes_global: 766
- active_code_ratio_global: 0.093505859375
- dead_code_ratio_global: 0.906494140625
- global_code_entropy_bits: 9.107343673706055
- global_code_perplexity: 551.5483968695353
- mean_per_image_used_codes: 52.333333333333336
- p50_per_image_used_codes: 55.0
- max_per_image_used_codes: 62
- mean_soft_perplexity_batch: 351.20945231119794
- mean_psnr_sem: 19.66388730208079
- mean_l1_sem: 0.07687549820790689
- mean_ms_ssim_sem: 0.6166008959213892
- mean_vq_mse: 0.10096408426761627
- latent_mean: -0.009143519525726637
- latent_std: 1.1610900561014812
- latent_min: -9.926763852437338
- latent_max: 10.90616257985433
- quantized_mean: -0.007286490018789967
- quantized_std: 1.1484249830245972
- quantized_min: -9.894265174865723
- quantized_max: 10.579552014668783

## Artifacts

- analysis: `results/analysis/stage1_semantic_vq/20260627_stage1_semantic_vq_1k_hardvq_codeinit_l1_kodak_analysis/analysis.json`
- top_codes: `results/analysis/stage1_semantic_vq/20260627_stage1_semantic_vq_1k_hardvq_codeinit_l1_kodak_analysis/top_codes.csv`
- output_dir: `results/analysis/stage1_semantic_vq/20260627_stage1_semantic_vq_1k_hardvq_codeinit_l1_kodak_analysis`
