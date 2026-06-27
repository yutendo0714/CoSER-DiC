# 20260627_stage3_residual_grid_diag_d32_b4_r025_clicproval41

Date: 2026-06-27T13:23:47

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/analyze_stage3_residual_grid_distribution.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --image-root /dpl/clic/professional/valid --crop-size 256 --max-images 64 --batch-size 8 --num-workers 4 --detail-downsample-factor 32 --detail-bits 4 --detail-range 0.25 --detail-prior outputs/stage3_residual_entropy/20260627_stage3_residual_poshuff_d32_b4_r025_4096calib_oi_div2k/static_residual_grid_position_huffman_prior.json --wandb-mode offline --run-name 20260627_stage3_residual_grid_diag_d32_b4_r025_clicproval41
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- image_roots: ['/dpl/clic/professional/valid']
- num_images: 41
- crop_size: 256
- detail_downsample_factor: 32
- detail_shape: [3, 8, 8]
- detail_bits: 4
- detail_range: 0.25
- detail_prior: outputs/stage3_residual_entropy/20260627_stage3_residual_poshuff_d32_b4_r025_4096calib_oi_div2k/static_residual_grid_position_huffman_prior.json
- total_symbols: 7872
- symbol_entropy_bits: 1.7128874063491821
- mean_huffman_bits_per_symbol: 1.7795985772357723
- symbol_counts: [0, 0, 0, 2, 9, 52, 506, 3256, 3415, 538, 85, 8, 1, 0, 0, 0]
- symbol_probs: [0.0, 0.0, 0.0, 0.0002540650311857462, 0.0011432926403358579, 0.006605691276490688, 0.06427845358848572, 0.413617879152298, 0.4338160455226898, 0.06834349781274796, 0.010797764174640179, 0.0010162601247429848, 0.0001270325155928731, 0.0, 0.0, 0.0]
- most_common_symbols: [(8, 3415, 0.4338160455226898), (7, 3256, 0.413617879152298), (9, 538, 0.06834349781274796), (6, 506, 0.06427845358848572), (10, 85, 0.010797764174640179), (5, 52, 0.006605691276490688), (4, 9, 0.0011432926403358579), (11, 8, 0.0010162601247429848)]
- least_common_nonzero_symbols: [(12, 1, 0.0001270325155928731), (3, 2, 0.0002540650311857462), (11, 8, 0.0010162601247429848), (4, 9, 0.0011432926403358579), (5, 52, 0.006605691276490688), (10, 85, 0.010797764174640179), (6, 506, 0.06427845358848572), (9, 538, 0.06834349781274796)]
- channel_entropy_bits: [1.7046356201171875, 1.7424687147140503, 1.6138538122177124]
- channel_abs_mean: [0.018571544674170756, 0.019426214841210244, 0.017384125480653094]
- channel_rms: [0.024928951672314834, 0.025704240806173072, 0.023296976179984944]
- channel_clip_ratio: [0.0, 0.0, 0.0]
- global_abs_mean: 0.018460628332011366
- global_rms: 0.024643389552824285
- global_clip_ratio: 0.0
- position_entropy_mean: 1.6002012491226196
- position_entropy_max: 2.060183525085449
- position_abs_mean_max: 0.024727352190671896
- position_rms_max: 0.036741040188998625

## Artifacts

- summary: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b4_r025_clicproval41/summary.json`
- diagnostics: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b4_r025_clicproval41/diagnostics.pt`
- abs_mean_map: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b4_r025_clicproval41/abs_mean_map.png`
- rms_map: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b4_r025_clicproval41/rms_map.png`
- entropy_map: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b4_r025_clicproval41/entropy_map.png`
- clip_ratio_map: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b4_r025_clicproval41/clip_ratio_map.png`
