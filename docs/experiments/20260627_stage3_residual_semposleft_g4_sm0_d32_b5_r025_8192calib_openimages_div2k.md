# 20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_8192calib_openimages_div2k

Date: 2026-06-27T15:21:06

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/fit_stage3_residual_huffman_prior.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --image-root /dpl/open-images-v6/train/data --image-root /dpl/div2k --crop-size 256 --max-images 8192 --batch-size 16 --num-workers 8 --detail-downsample-factor 32 --detail-bits 5 --detail-range 0.25 --coding-mode semantic_position_leftctx_huffman --semantic-group-count 4 --semantic-group-iters 25 --smoothing-count 0 --wandb-mode offline --run-name 20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_8192calib_openimages_div2k
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- prior: outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_8192calib_openimages_div2k/static_residual_grid_semantic_position_leftctx_huffman_prior.json
- num_images: 8192
- crop_size: 256
- detail_downsample_factor: 32
- detail_shape: [3, 8, 8]
- detail_bits: 5
- detail_range: 0.25
- coding_mode: semantic_position_leftctx_huffman
- smoothing_count: 0
- semantic_group_count: 4
- left_context_count: 4
- total_symbols: 1572864
- counts: [117, 40, 60, 65, 128, 290, 431, 857, 1702, 3357, 7292, 16046, 37049, 85976, 198830, 399897, 402626, 222272, 106083, 48586, 21901, 9892, 4579, 2258, 1099, 519, 322, 175, 104, 81, 69, 161]
- symbol_probs: [7.43865966796875e-05, 2.5431314497836865e-05, 3.814697265625e-05, 4.132588583161123e-05, 8.138021075865254e-05, 0.00018437702965456992, 0.00027402242994867265, 0.0005448659067042172, 0.0010821024188771844, 0.0021343231201171875, 0.0046361288987100124, 0.0102017717435956, 0.023555120453238487, 0.0546620674431324, 0.12641270458698273, 0.25424766540527344, 0.25598272681236267, 0.1413167268037796, 0.06744575500488281, 0.030890146270394325, 0.013924281112849712, 0.0062891640700399876, 0.0029112498741596937, 0.0014355977764353156, 0.0006987254018895328, 0.0003299713134765625, 0.00020472209143918008, 0.00011126200115540996, 6.612142169615254e-05, 5.14984130859375e-05, 4.38690185546875e-05, 0.00010236104571959004]
- symbol_entropy_bits: 2.8924365043640137
- mean_huffman_bits_per_symbol: 2.697382609049479
- fixed_bits_per_symbol: 5
- mean_residual_abs: 0.02178144212666666
- residual_std: 0.03047531290434756
- clip_ratio: 0.000156402587890625

## Artifacts

- prior: `outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_8192calib_openimages_div2k/static_residual_grid_semantic_position_leftctx_huffman_prior.json`
- summary: `outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_8192calib_openimages_div2k/summary.json`
