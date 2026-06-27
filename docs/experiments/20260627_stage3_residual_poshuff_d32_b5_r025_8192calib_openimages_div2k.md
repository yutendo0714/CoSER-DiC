# 20260627_stage3_residual_poshuff_d32_b5_r025_8192calib_openimages_div2k

Date: 2026-06-27T13:26:19

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/fit_stage3_residual_huffman_prior.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --image-root /dpl/open-images-v6/train/data --image-root /dpl/div2k --crop-size 256 --max-images 8192 --batch-size 16 --num-workers 8 --detail-downsample-factor 32 --detail-bits 5 --detail-range 0.25 --coding-mode position_huffman --smoothing-count 1 --wandb-mode offline --run-name 20260627_stage3_residual_poshuff_d32_b5_r025_8192calib_openimages_div2k
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- prior: outputs/stage3_residual_entropy/20260627_stage3_residual_poshuff_d32_b5_r025_8192calib_openimages_div2k/static_residual_grid_position_huffman_prior.json
- num_images: 8192
- crop_size: 256
- detail_downsample_factor: 32
- detail_shape: [3, 8, 8]
- detail_bits: 5
- detail_range: 0.25
- coding_mode: position_huffman
- smoothing_count: 1
- total_symbols: 1572864
- counts: [117, 40, 60, 65, 128, 290, 431, 857, 1702, 3357, 7291, 16047, 37048, 85969, 198834, 399887, 402625, 222278, 106092, 48585, 21905, 9891, 4576, 2259, 1099, 519, 322, 175, 104, 81, 69, 161]
- symbol_probs: [7.43865966796875e-05, 2.5431314497836865e-05, 3.814697265625e-05, 4.132588583161123e-05, 8.138021075865254e-05, 0.00018437702965456992, 0.00027402242994867265, 0.0005448659067042172, 0.0010821024188771844, 0.0021343231201171875, 0.00463549280539155, 0.010202407836914062, 0.02355448342859745, 0.05465761944651604, 0.12641525268554688, 0.25424131751060486, 0.25598207116127014, 0.14132054150104523, 0.06745147705078125, 0.030889511108398438, 0.013926823623478413, 0.0062885284423828125, 0.0029093425255268812, 0.0014362335205078125, 0.0006987254018895328, 0.0003299713134765625, 0.00020472209143918008, 0.00011126200115540996, 6.612142169615254e-05, 5.14984130859375e-05, 4.38690185546875e-05, 0.00010236104571959004]
- symbol_entropy_bits: 2.8924384117126465
- mean_huffman_bits_per_symbol: 2.896092096964518
- fixed_bits_per_symbol: 5
- mean_residual_abs: 0.021781434043077752
- residual_std: 0.03047513046705603
- clip_ratio: 0.000156402587890625

## Artifacts

- prior: `outputs/stage3_residual_entropy/20260627_stage3_residual_poshuff_d32_b5_r025_8192calib_openimages_div2k/static_residual_grid_position_huffman_prior.json`
- summary: `outputs/stage3_residual_entropy/20260627_stage3_residual_poshuff_d32_b5_r025_8192calib_openimages_div2k/summary.json`
