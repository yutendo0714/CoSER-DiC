# 20260627_stage3_residual_grid_diag_d32_b4_r025_div2k100

Date: 2026-06-27T13:23:16

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/analyze_stage3_residual_grid_distribution.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --image-root /dpl/div2k --crop-size 256 --max-images 100 --batch-size 8 --num-workers 4 --detail-downsample-factor 32 --detail-bits 4 --detail-range 0.25 --detail-prior outputs/stage3_residual_entropy/20260627_stage3_residual_poshuff_d32_b4_r025_4096calib_oi_div2k/static_residual_grid_position_huffman_prior.json --wandb-mode offline --run-name 20260627_stage3_residual_grid_diag_d32_b4_r025_div2k100
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- image_roots: ['/dpl/div2k']
- num_images: 100
- crop_size: 256
- detail_downsample_factor: 32
- detail_shape: [3, 8, 8]
- detail_bits: 4
- detail_range: 0.25
- detail_prior: outputs/stage3_residual_entropy/20260627_stage3_residual_poshuff_d32_b4_r025_4096calib_oi_div2k/static_residual_grid_position_huffman_prior.json
- total_symbols: 19200
- symbol_entropy_bits: 1.9304943084716797
- mean_huffman_bits_per_symbol: 1.97515625
- symbol_counts: [0, 0, 2, 13, 24, 192, 1407, 7568, 7531, 1894, 448, 91, 26, 4, 0, 0]
- symbol_probs: [0.0, 0.0, 0.00010416666918899864, 0.0006770833279006183, 0.0012499999720603228, 0.009999999776482582, 0.07328125089406967, 0.3941666781902313, 0.3922395706176758, 0.09864583611488342, 0.023333333432674408, 0.004739583469927311, 0.0013541666558012366, 0.00020833333837799728, 0.0, 0.0]
- most_common_symbols: [(7, 7568, 0.3941666781902313), (8, 7531, 0.3922395706176758), (9, 1894, 0.09864583611488342), (6, 1407, 0.07328125089406967), (10, 448, 0.023333333432674408), (5, 192, 0.009999999776482582), (11, 91, 0.004739583469927311), (12, 26, 0.0013541666558012366)]
- least_common_nonzero_symbols: [(2, 2, 0.00010416666918899864), (13, 4, 0.00020833333837799728), (3, 13, 0.0006770833279006183), (4, 24, 0.0012499999720603228), (12, 26, 0.0013541666558012366), (11, 91, 0.004739583469927311), (5, 192, 0.009999999776482582), (10, 448, 0.023333333432674408)]
- channel_entropy_bits: [1.8906325101852417, 1.9284417629241943, 1.9230132102966309]
- channel_abs_mean: [0.021323041621944865, 0.022125777287001255, 0.021667727884778283]
- channel_rms: [0.02901290842484448, 0.030596636962739024, 0.03055757466894297]
- channel_clip_ratio: [0.0, 0.0, 0.0]
- global_abs_mean: 0.021705515597908138
- global_rms: 0.03005570668550882
- global_clip_ratio: 0.0
- position_entropy_mean: 1.8617902994155884
- position_entropy_max: 2.201828956604004
- position_abs_mean_max: 0.02788899779319763
- position_rms_max: 0.04118255181714546

## Artifacts

- summary: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b4_r025_div2k100/summary.json`
- diagnostics: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b4_r025_div2k100/diagnostics.pt`
- abs_mean_map: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b4_r025_div2k100/abs_mean_map.png`
- rms_map: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b4_r025_div2k100/rms_map.png`
- entropy_map: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b4_r025_div2k100/entropy_map.png`
- clip_ratio_map: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b4_r025_div2k100/clip_ratio_map.png`
