# 20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_oi_div2k

Date: 2026-06-27T12:24:41

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/fit_stage3_residual_huffman_prior.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --image-root /dpl/open-images-v6/train/data --image-root /dpl/div2k --max-images 4096 --batch-size 16 --num-workers 8 --detail-downsample-factor 32 --detail-bits 5 --detail-range 0.25 --coding-mode position_huffman --smoothing-count 1 --wandb-mode offline --run-name 20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_oi_div2k
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- prior: outputs/stage3_residual_entropy/20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_oi_div2k/static_residual_grid_position_huffman_prior.json
- num_images: 4096
- crop_size: 256
- detail_downsample_factor: 32
- detail_shape: [3, 8, 8]
- detail_bits: 5
- detail_range: 0.25
- coding_mode: position_huffman
- smoothing_count: 1
- total_symbols: 786432
- counts: [60, 20, 27, 33, 70, 159, 240, 462, 840, 1712, 3650, 7978, 18717, 42646, 99357, 200266, 201110, 111139, 53030, 24153, 11016, 4990, 2334, 1157, 547, 245, 178, 100, 52, 40, 38, 66]
- symbol_probs: [7.62939453125e-05, 2.5431314497836865e-05, 3.4332275390625e-05, 4.1961669921875e-05, 8.900960528990254e-05, 0.000202178955078125, 0.00030517578125, 0.00058746337890625, 0.001068115234375, 0.0021769206505268812, 0.0046412148512899876, 0.010144551284611225, 0.023799896240234375, 0.05422719195485115, 0.12633895874023438, 0.25465139746665955, 0.25572457909584045, 0.14132054150104523, 0.06743112951517105, 0.030712127685546875, 0.014007568359375, 0.0063451132737100124, 0.00296783447265625, 0.0014712015399709344, 0.0006955464486964047, 0.00031153360032476485, 0.00022633869957644492, 0.00012715657067019492, 6.612142169615254e-05, 5.086262899567373e-05, 4.831949991057627e-05, 8.392333984375e-05]
- symbol_entropy_bits: 2.8938543796539307
- mean_huffman_bits_per_symbol: 2.898509979248047
- fixed_bits_per_symbol: 5
- mean_residual_abs: 0.021804466067502897
- residual_std: 0.030544521252252783
- clip_ratio: 0.00013605753580729166

## Artifacts

- prior: `outputs/stage3_residual_entropy/20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_oi_div2k/static_residual_grid_position_huffman_prior.json`
- summary: `outputs/stage3_residual_entropy/20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_oi_div2k/summary.json`
