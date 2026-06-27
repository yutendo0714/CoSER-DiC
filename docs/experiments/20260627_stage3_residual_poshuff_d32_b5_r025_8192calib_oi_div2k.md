# 20260627_stage3_residual_poshuff_d32_b5_r025_8192calib_oi_div2k

Date: 2026-06-27T13:24:52

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/fit_stage3_residual_huffman_prior.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --config configs/train/train_stage1_semantic_vq.yaml --crop-size 256 --max-images 8192 --batch-size 16 --num-workers 8 --detail-downsample-factor 32 --detail-bits 5 --detail-range 0.25 --coding-mode position_huffman --smoothing-count 1 --wandb-mode offline --run-name 20260627_stage3_residual_poshuff_d32_b5_r025_8192calib_oi_div2k
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- prior: outputs/stage3_residual_entropy/20260627_stage3_residual_poshuff_d32_b5_r025_8192calib_oi_div2k/static_residual_grid_position_huffman_prior.json
- num_images: 256
- crop_size: 256
- detail_downsample_factor: 32
- detail_shape: [3, 8, 8]
- detail_bits: 5
- detail_range: 0.25
- coding_mode: position_huffman
- smoothing_count: 1
- total_symbols: 49152
- counts: [0, 0, 0, 0, 1, 2, 1, 9, 25, 75, 177, 439, 1097, 2720, 6438, 13657, 12913, 6470, 2893, 1261, 545, 236, 107, 48, 18, 10, 8, 1, 1, 0, 0, 0]
- symbol_probs: [0.0, 0.0, 0.0, 0.0, 2.0345052689663135e-05, 4.069010537932627e-05, 2.0345052689663135e-05, 0.00018310546875, 0.0005086262826807797, 0.00152587890625, 0.00360107421875, 0.008931477554142475, 0.02231852151453495, 0.0553385429084301, 0.1309814453125, 0.2778523862361908, 0.2627156674861908, 0.1316324919462204, 0.0588582344353199, 0.02565511129796505, 0.011088053695857525, 0.0048014321364462376, 0.0021769206505268812, 0.0009765625, 0.0003662109375, 0.00020345051598269492, 0.00016276042151730508, 2.0345052689663135e-05, 2.0345052689663135e-05, 0.0, 0.0, 0.0]
- symbol_entropy_bits: 2.779531478881836
- mean_huffman_bits_per_symbol: 2.85308837890625
- fixed_bits_per_symbol: 5
- mean_residual_abs: 0.02004766425428291
- residual_std: 0.027638285510911985
- clip_ratio: 0.0

## Artifacts

- prior: `outputs/stage3_residual_entropy/20260627_stage3_residual_poshuff_d32_b5_r025_8192calib_oi_div2k/static_residual_grid_position_huffman_prior.json`
- summary: `outputs/stage3_residual_entropy/20260627_stage3_residual_poshuff_d32_b5_r025_8192calib_oi_div2k/summary.json`
