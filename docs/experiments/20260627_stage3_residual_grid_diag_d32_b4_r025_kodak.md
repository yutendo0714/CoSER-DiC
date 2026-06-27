# 20260627_stage3_residual_grid_diag_d32_b4_r025_kodak

Date: 2026-06-27T12:21:18

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/analyze_stage3_residual_grid_distribution.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --image-root /dpl/kodak --max-images 24 --batch-size 4 --num-workers 4 --detail-downsample-factor 32 --detail-bits 4 --detail-range 0.25 --detail-prior outputs/stage3_residual_entropy/20260627_stage3_residual_huffman_d32_b4_r025_4096calib_oi_div2k/static_residual_grid_huffman_prior.json --wandb-mode offline --run-name 20260627_stage3_residual_grid_diag_d32_b4_r025_kodak
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- image_roots: ['/dpl/kodak']
- num_images: 24
- crop_size: 256
- detail_downsample_factor: 32
- detail_shape: [3, 8, 8]
- detail_bits: 4
- detail_range: 0.25
- detail_prior: outputs/stage3_residual_entropy/20260627_stage3_residual_huffman_d32_b4_r025_4096calib_oi_div2k/static_residual_grid_huffman_prior.json
- total_symbols: 4608
- symbol_entropy_bits: 1.8214513063430786
- mean_huffman_bits_per_symbol: 1.8756510416666667
- symbol_counts: [0, 0, 0, 0, 6, 48, 265, 1649, 2097, 453, 75, 11, 3, 1, 0, 0]
- symbol_probs: [0.0, 0.0, 0.0, 0.0, 0.0013020833721384406, 0.010416666977107525, 0.0575086809694767, 0.3578559160232544, 0.455078125, 0.0983072891831398, 0.01627604104578495, 0.0023871527519077063, 0.0006510416860692203, 0.00021701389050576836, 0.0, 0.0]
- most_common_symbols: [(8, 2097, 0.455078125), (7, 1649, 0.3578559160232544), (9, 453, 0.0983072891831398), (6, 265, 0.0575086809694767), (10, 75, 0.01627604104578495), (5, 48, 0.010416666977107525), (11, 11, 0.0023871527519077063), (4, 6, 0.0013020833721384406)]
- least_common_nonzero_symbols: [(13, 1, 0.00021701389050576836), (12, 3, 0.0006510416860692203), (4, 6, 0.0013020833721384406), (11, 11, 0.0023871527519077063), (5, 48, 0.010416666977107525), (10, 75, 0.01627604104578495), (6, 265, 0.0575086809694767), (9, 453, 0.0983072891831398)]
- channel_entropy_bits: [1.8364732265472412, 1.761292576789856, 1.7426097393035889]
- channel_abs_mean: [0.02064629746503973, 0.021767377804280837, 0.018249355313552467]
- channel_rms: [0.027402895823187626, 0.028966796283693023, 0.024719162170255504]
- channel_clip_ratio: [0.0, 0.0, 0.0]
- global_abs_mean: 0.02022101019429101
- global_rms: 0.027029618092378718
- global_clip_ratio: 0.0
- position_entropy_mean: 1.6279958486557007
- position_entropy_max: 2.2295737266540527
- position_abs_mean_max: 0.031221737464269
- position_rms_max: 0.04380922859394903

## Artifacts

- summary: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b4_r025_kodak/summary.json`
- diagnostics: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b4_r025_kodak/diagnostics.pt`
- abs_mean_map: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b4_r025_kodak/abs_mean_map.png`
- rms_map: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b4_r025_kodak/rms_map.png`
- entropy_map: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b4_r025_kodak/entropy_map.png`
- clip_ratio_map: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b4_r025_kodak/clip_ratio_map.png`
