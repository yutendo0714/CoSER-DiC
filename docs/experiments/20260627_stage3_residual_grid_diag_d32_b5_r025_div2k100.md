# 20260627_stage3_residual_grid_diag_d32_b5_r025_div2k100

Date: 2026-06-27T13:22:59

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/analyze_stage3_residual_grid_distribution.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --image-root /dpl/div2k --crop-size 256 --max-images 100 --batch-size 8 --num-workers 4 --detail-downsample-factor 32 --detail-bits 5 --detail-range 0.25 --detail-prior outputs/stage3_residual_entropy/20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_oi_div2k/static_residual_grid_position_huffman_prior.json --wandb-mode offline --run-name 20260627_stage3_residual_grid_diag_d32_b5_r025_div2k100
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- image_roots: ['/dpl/div2k']
- num_images: 100
- crop_size: 256
- detail_downsample_factor: 32
- detail_shape: [3, 8, 8]
- detail_bits: 5
- detail_range: 0.25
- detail_prior: outputs/stage3_residual_entropy/20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_oi_div2k/static_residual_grid_position_huffman_prior.json
- total_symbols: 19200
- symbol_entropy_bits: 2.8850820064544678
- mean_huffman_bits_per_symbol: 2.90078125
- symbol_counts: [0, 0, 0, 0, 0, 6, 6, 3, 10, 21, 55, 157, 429, 1059, 2391, 5071, 4837, 2573, 1256, 699, 336, 146, 77, 30, 19, 13, 6, 0, 0, 0, 0, 0]
- symbol_probs: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0003124999930150807, 0.0003124999930150807, 0.00015624999650754035, 0.0005208333604969084, 0.0010937500046566129, 0.002864583395421505, 0.008177082985639572, 0.022343749180436134, 0.055156249552965164, 0.12453124672174454, 0.2641145884990692, 0.25192707777023315, 0.1340104192495346, 0.06541666388511658, 0.036406248807907104, 0.017500000074505806, 0.007604166865348816, 0.004010416567325592, 0.0015625000232830644, 0.000989583320915699, 0.0006770833279006183, 0.0003124999930150807, 0.0, 0.0, 0.0, 0.0, 0.0]
- most_common_symbols: [(15, 5071, 0.2641145884990692), (16, 4837, 0.25192707777023315), (17, 2573, 0.1340104192495346), (14, 2391, 0.12453124672174454), (18, 1256, 0.06541666388511658), (13, 1059, 0.055156249552965164), (19, 699, 0.036406248807907104), (12, 429, 0.022343749180436134)]
- least_common_nonzero_symbols: [(7, 3, 0.00015624999650754035), (5, 6, 0.0003124999930150807), (6, 6, 0.0003124999930150807), (26, 6, 0.0003124999930150807), (8, 10, 0.0005208333604969084), (25, 13, 0.0006770833279006183), (24, 19, 0.000989583320915699), (9, 21, 0.0010937500046566129)]
- channel_entropy_bits: [2.8455402851104736, 2.8816182613372803, 2.864647150039673]
- channel_abs_mean: [0.021323344248812645, 0.022128798173798717, 0.021664446861541363]
- channel_rms: [0.029013277717720866, 0.03059930321306826, 0.030552021416746407]
- channel_clip_ratio: [0.0, 0.0, 0.0]
- global_abs_mean: 0.021705529761384245
- global_rms: 0.030054867449178513
- global_clip_ratio: 0.0
- position_entropy_mean: 2.758890390396118
- position_entropy_max: 3.0952916145324707
- position_abs_mean_max: 0.027889095321297644
- position_rms_max: 0.04117871093370041

## Artifacts

- summary: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b5_r025_div2k100/summary.json`
- diagnostics: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b5_r025_div2k100/diagnostics.pt`
- abs_mean_map: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b5_r025_div2k100/abs_mean_map.png`
- rms_map: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b5_r025_div2k100/rms_map.png`
- entropy_map: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b5_r025_div2k100/entropy_map.png`
- clip_ratio_map: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b5_r025_div2k100/clip_ratio_map.png`
