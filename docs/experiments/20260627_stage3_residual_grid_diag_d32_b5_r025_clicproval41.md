# 20260627_stage3_residual_grid_diag_d32_b5_r025_clicproval41

Date: 2026-06-27T13:23:32

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/analyze_stage3_residual_grid_distribution.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --image-root /dpl/clic/professional/valid --crop-size 256 --max-images 64 --batch-size 8 --num-workers 4 --detail-downsample-factor 32 --detail-bits 5 --detail-range 0.25 --detail-prior outputs/stage3_residual_entropy/20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_oi_div2k/static_residual_grid_position_huffman_prior.json --wandb-mode offline --run-name 20260627_stage3_residual_grid_diag_d32_b5_r025_clicproval41
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- image_roots: ['/dpl/clic/professional/valid']
- num_images: 41
- crop_size: 256
- detail_downsample_factor: 32
- detail_shape: [3, 8, 8]
- detail_bits: 5
- detail_range: 0.25
- detail_prior: outputs/stage3_residual_entropy/20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_oi_div2k/static_residual_grid_position_huffman_prior.json
- total_symbols: 7872
- symbol_entropy_bits: 2.6534905433654785
- mean_huffman_bits_per_symbol: 2.7019817073170733
- symbol_counts: [0, 0, 0, 0, 0, 0, 0, 2, 2, 10, 16, 41, 139, 391, 1064, 2160, 2200, 1179, 405, 164, 65, 23, 8, 2, 0, 1, 0, 0, 0, 0, 0, 0]
- symbol_probs: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0002540650311857462, 0.0002540650311857462, 0.001270325155928731, 0.0020325202494859695, 0.0052083334885537624, 0.01765752024948597, 0.049669716507196426, 0.1351626068353653, 0.27439025044441223, 0.2794715464115143, 0.14977134764194489, 0.051448170095682144, 0.02083333395421505, 0.00825711339712143, 0.002921747975051403, 0.0010162601247429848, 0.0002540650311857462, 0.0, 0.0001270325155928731, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
- most_common_symbols: [(16, 2200, 0.2794715464115143), (15, 2160, 0.27439025044441223), (17, 1179, 0.14977134764194489), (14, 1064, 0.1351626068353653), (18, 405, 0.051448170095682144), (13, 391, 0.049669716507196426), (19, 164, 0.02083333395421505), (12, 139, 0.01765752024948597)]
- least_common_nonzero_symbols: [(25, 1, 0.0001270325155928731), (7, 2, 0.0002540650311857462), (8, 2, 0.0002540650311857462), (23, 2, 0.0002540650311857462), (22, 8, 0.0010162601247429848), (9, 10, 0.001270325155928731), (10, 16, 0.0020325202494859695), (21, 23, 0.002921747975051403)]
- channel_entropy_bits: [2.626847743988037, 2.6973142623901367, 2.5516128540039062]
- channel_abs_mean: [0.018571096322919, 0.019424922863081298, 0.017387912287087998]
- channel_rms: [0.024928188427313792, 0.025703001722653794, 0.023301448992603828]
- channel_clip_ratio: [0.0, 0.0, 0.0]
- global_abs_mean: 0.018461310491029434
- global_rms: 0.024644213047523802
- global_clip_ratio: 0.0
- position_entropy_mean: 2.449101686477661
- position_entropy_max: 2.8883981704711914
- position_abs_mean_max: 0.024729989287329883
- position_rms_max: 0.03674462138793235

## Artifacts

- summary: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b5_r025_clicproval41/summary.json`
- diagnostics: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b5_r025_clicproval41/diagnostics.pt`
- abs_mean_map: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b5_r025_clicproval41/abs_mean_map.png`
- rms_map: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b5_r025_clicproval41/rms_map.png`
- entropy_map: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b5_r025_clicproval41/entropy_map.png`
- clip_ratio_map: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b5_r025_clicproval41/clip_ratio_map.png`
