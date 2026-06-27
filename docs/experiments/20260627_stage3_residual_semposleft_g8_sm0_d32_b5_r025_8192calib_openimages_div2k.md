# 20260627_stage3_residual_semposleft_g8_sm0_d32_b5_r025_8192calib_openimages_div2k

Date: 2026-06-27T15:35:23

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/fit_stage3_residual_huffman_prior.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --image-root /dpl/open-images-v6/train/data --image-root /dpl/div2k --crop-size 256 --max-images 8192 --batch-size 16 --num-workers 8 --detail-downsample-factor 32 --detail-bits 5 --detail-range 0.25 --coding-mode semantic_position_leftctx_huffman --semantic-group-count 8 --semantic-group-iters 25 --smoothing-count 0 --wandb-mode offline --run-name 20260627_stage3_residual_semposleft_g8_sm0_d32_b5_r025_8192calib_openimages_div2k
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- prior: outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g8_sm0_d32_b5_r025_8192calib_openimages_div2k/static_residual_grid_semantic_position_leftctx_huffman_prior.json
- num_images: 8192
- crop_size: 256
- detail_downsample_factor: 32
- detail_shape: [3, 8, 8]
- detail_bits: 5
- detail_range: 0.25
- coding_mode: semantic_position_leftctx_huffman
- smoothing_count: 0
- semantic_group_count: 8
- left_context_count: 4
- total_symbols: 1572864
- counts: [117, 40, 60, 65, 128, 290, 431, 856, 1702, 3358, 7291, 16043, 37045, 85972, 198832, 399893, 402635, 222272, 106087, 48588, 21900, 9892, 4577, 2260, 1099, 519, 322, 175, 104, 81, 69, 161]
- symbol_probs: [7.43865966796875e-05, 2.5431314497836865e-05, 3.814697265625e-05, 4.132588583161123e-05, 8.138021075865254e-05, 0.00018437702965456992, 0.00027402242994867265, 0.0005442301626317203, 0.0010821024188771844, 0.0021349589806050062, 0.00463549280539155, 0.010199864394962788, 0.023552576079964638, 0.05465952679514885, 0.1264139860868454, 0.2542451322078705, 0.2559884488582611, 0.1413167268037796, 0.06744829565286636, 0.03089141845703125, 0.01392364501953125, 0.0062891640700399876, 0.0029099781531840563, 0.0014368692645803094, 0.0006987254018895328, 0.0003299713134765625, 0.00020472209143918008, 0.00011126200115540996, 6.612142169615254e-05, 5.14984130859375e-05, 4.38690185546875e-05, 0.00010236104571959004]
- symbol_entropy_bits: 2.8924169540405273
- mean_huffman_bits_per_symbol: 2.6729749043782554
- fixed_bits_per_symbol: 5
- mean_residual_abs: 0.021781172945338767
- residual_std: 0.030475003324601624
- clip_ratio: 0.000156402587890625

## Artifacts

- prior: `outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g8_sm0_d32_b5_r025_8192calib_openimages_div2k/static_residual_grid_semantic_position_leftctx_huffman_prior.json`
- summary: `outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g8_sm0_d32_b5_r025_8192calib_openimages_div2k/summary.json`
