# 20260627_stage1_semantic_vq_3k_hardvq_from_5k_ae_codeinit_vq010_usage001_kodak_analysis

Date: 2026-06-27T03:31:07

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/analyze_stage1_semantic_vq.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_3k_hardvq_from_5k_ae_codeinit_vq010_usage001.pt --image-root /dpl/kodak --batch-size 4 --num-workers 4 --quantize-mix 1 --wandb-mode offline --run-name 20260627_stage1_semantic_vq_3k_hardvq_from_5k_ae_codeinit_vq010_usage001_kodak_analysis
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_3k_hardvq_from_5k_ae_codeinit_vq010_usage001.pt
- quantize_mix: 1.0
- num_images: 24
- total_tokens: 1536
- active_codes_global: 1022
- active_code_ratio_global: 0.124755859375
- dead_code_ratio_global: 0.875244140625
- global_code_entropy_bits: 9.7420072555542
- global_code_perplexity: 856.3206200805207
- mean_per_image_used_codes: 56.125
- p50_per_image_used_codes: 57.0
- max_per_image_used_codes: 64
- mean_soft_perplexity_batch: 222.8650156656901
- mean_psnr_sem: 20.3733553091685
- mean_l1_sem: 0.06961414761220415
- mean_ms_ssim_sem: 0.663416363298893
- mean_vq_mse: 0.5324057290951411
- latent_mean: 0.0016129417344927788
- latent_std: 1.8507709105809529
- latent_min: -22.96240520477295
- latent_max: 25.377763112386067
- quantized_mean: -0.003761595735947291
- quantized_std: 1.7487040360768635
- quantized_min: -22.08959706624349
- quantized_max: 24.508031845092773

## Artifacts

- analysis: `results/analysis/stage1_semantic_vq/20260627_stage1_semantic_vq_3k_hardvq_from_5k_ae_codeinit_vq010_usage001_kodak_analysis/analysis.json`
- top_codes: `results/analysis/stage1_semantic_vq/20260627_stage1_semantic_vq_3k_hardvq_from_5k_ae_codeinit_vq010_usage001_kodak_analysis/top_codes.csv`
- output_dir: `results/analysis/stage1_semantic_vq/20260627_stage1_semantic_vq_3k_hardvq_from_5k_ae_codeinit_vq010_usage001_kodak_analysis`
