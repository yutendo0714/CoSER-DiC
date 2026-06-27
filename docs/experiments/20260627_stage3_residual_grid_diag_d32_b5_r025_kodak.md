# 20260627_stage3_residual_grid_diag_d32_b5_r025_kodak

Date: 2026-06-27T12:21:04

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/analyze_stage3_residual_grid_distribution.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --image-root /dpl/kodak --max-images 24 --batch-size 4 --num-workers 4 --detail-downsample-factor 32 --detail-bits 5 --detail-range 0.25 --detail-prior outputs/stage3_residual_entropy/20260627_stage3_residual_huffman_d32_b5_r025_4096calib_oi_div2k/static_residual_grid_huffman_prior.json --wandb-mode offline --run-name 20260627_stage3_residual_grid_diag_d32_b5_r025_kodak
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- image_roots: ['/dpl/kodak']
- num_images: 24
- crop_size: 256
- detail_downsample_factor: 32
- detail_shape: [3, 8, 8]
- detail_bits: 5
- detail_range: 0.25
- detail_prior: outputs/stage3_residual_entropy/20260627_stage3_residual_huffman_d32_b5_r025_4096calib_oi_div2k/static_residual_grid_huffman_prior.json
- total_symbols: 4608
- symbol_entropy_bits: 2.772547960281372
- mean_huffman_bits_per_symbol: 2.810546875
- symbol_counts: [0, 0, 0, 0, 0, 0, 0, 0, 2, 5, 21, 31, 81, 194, 529, 1105, 1337, 725, 332, 147, 55, 27, 10, 2, 3, 1, 1, 0, 0, 0, 0, 0]
- symbol_probs: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0004340277810115367, 0.0010850694961845875, 0.0045572915114462376, 0.0067274305038154125, 0.017578125, 0.0421006940305233, 0.1148003488779068, 0.2398003488779068, 0.2901475727558136, 0.1573350727558136, 0.0720486119389534, 0.0319010429084301, 0.011935763992369175, 0.005859375, 0.002170138992369175, 0.0004340277810115367, 0.0006510416860692203, 0.00021701389050576836, 0.00021701389050576836, 0.0, 0.0, 0.0, 0.0, 0.0]
- most_common_symbols: [(16, 1337, 0.2901475727558136), (15, 1105, 0.2398003488779068), (17, 725, 0.1573350727558136), (14, 529, 0.1148003488779068), (18, 332, 0.0720486119389534), (13, 194, 0.0421006940305233), (19, 147, 0.0319010429084301), (12, 81, 0.017578125)]
- least_common_nonzero_symbols: [(25, 1, 0.00021701389050576836), (26, 1, 0.00021701389050576836), (8, 2, 0.0004340277810115367), (23, 2, 0.0004340277810115367), (24, 3, 0.0006510416860692203), (9, 5, 0.0010850694961845875), (22, 10, 0.002170138992369175), (10, 21, 0.0045572915114462376)]
- channel_entropy_bits: [2.8013100624084473, 2.7075283527374268, 2.6556482315063477]
- channel_abs_mean: [0.02064629746503973, 0.021767377804280837, 0.018249355313552467]
- channel_rms: [0.027402895823187626, 0.028966796283693023, 0.024719162170255504]
- channel_clip_ratio: [0.0, 0.0, 0.0]
- global_abs_mean: 0.02022101019429101
- global_rms: 0.027029618092378718
- global_clip_ratio: 0.0
- position_entropy_mean: 2.4220592975616455
- position_entropy_max: 3.0178940296173096
- position_abs_mean_max: 0.031221737464269
- position_rms_max: 0.04380922859394903

## Artifacts

- summary: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b5_r025_kodak/summary.json`
- diagnostics: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b5_r025_kodak/diagnostics.pt`
- abs_mean_map: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b5_r025_kodak/abs_mean_map.png`
- rms_map: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b5_r025_kodak/rms_map.png`
- entropy_map: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b5_r025_kodak/entropy_map.png`
- clip_ratio_map: `results/analysis/stage3_residual_grid_distribution/20260627_stage3_residual_grid_diag_d32_b5_r025_kodak/clip_ratio_map.png`
