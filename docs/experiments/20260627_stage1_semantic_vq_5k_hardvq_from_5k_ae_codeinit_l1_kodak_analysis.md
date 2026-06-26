# 20260627_stage1_semantic_vq_5k_hardvq_from_5k_ae_codeinit_l1_kodak_analysis

Date: 2026-06-27T03:20:52

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/analyze_stage1_semantic_vq.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_l1.pt --image-root /dpl/kodak --batch-size 4 --num-workers 4 --quantize-mix 1 --wandb-mode offline --run-name 20260627_stage1_semantic_vq_5k_hardvq_from_5k_ae_codeinit_l1_kodak_analysis
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_l1.pt
- quantize_mix: 1.0
- num_images: 24
- total_tokens: 1536
- active_codes_global: 729
- active_code_ratio_global: 0.0889892578125
- dead_code_ratio_global: 0.9110107421875
- global_code_entropy_bits: 9.04190444946289
- global_code_perplexity: 527.089612108577
- mean_per_image_used_codes: 53.083333333333336
- p50_per_image_used_codes: 55.0
- max_per_image_used_codes: 61
- mean_soft_perplexity_batch: 199.6355972290039
- mean_psnr_sem: 20.122823476791382
- mean_l1_sem: 0.07248667037735383
- mean_ms_ssim_sem: 0.6373254644374052
- mean_vq_mse: 0.21574736634890238
- latent_mean: -0.05024346926560005
- latent_std: 1.2492499351501465
- latent_min: -15.624294916788736
- latent_max: 17.201356252034504
- quantized_mean: -0.056390068804224334
- quantized_std: 1.1986231009165447
- quantized_min: -15.252808888753256
- quantized_max: 16.672696590423584

## Artifacts

- analysis: `results/analysis/stage1_semantic_vq/20260627_stage1_semantic_vq_5k_hardvq_from_5k_ae_codeinit_l1_kodak_analysis/analysis.json`
- top_codes: `results/analysis/stage1_semantic_vq/20260627_stage1_semantic_vq_5k_hardvq_from_5k_ae_codeinit_l1_kodak_analysis/top_codes.csv`
- output_dir: `results/analysis/stage1_semantic_vq/20260627_stage1_semantic_vq_5k_hardvq_from_5k_ae_codeinit_l1_kodak_analysis`
