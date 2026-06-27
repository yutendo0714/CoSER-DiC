# 20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_4096calib_stage1_lpips002_ft500

Date: 2026-06-27T18:10:39

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/fit_stage3_residual_huffman_prior.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_active5k_lpips002_fp32loss_ft500.pt --image-root /dpl/open-images-v6/train/data --image-root /dpl/div2k --max-images 4096 --batch-size 16 --num-workers 8 --detail-downsample-factor 32 --detail-bits 5 --detail-range 0.25 --coding-mode semantic_position_leftctx_huffman --smoothing-count 0 --semantic-group-count 4 --wandb-mode offline --run-name 20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_4096calib_stage1_lpips002_ft500
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_active5k_lpips002_fp32loss_ft500.pt
- prior: outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_4096calib_stage1_lpips002_ft500/static_residual_grid_semantic_position_leftctx_huffman_prior.json
- num_images: 4096
- crop_size: 256
- detail_downsample_factor: 32
- detail_shape: [3, 8, 8]
- detail_bits: 5
- detail_range: 0.25
- coding_mode: semantic_position_leftctx_huffman
- smoothing_count: 0
- semantic_group_count: 4
- left_context_count: 4
- total_symbols: 786432
- counts: [48, 13, 27, 44, 63, 125, 196, 327, 751, 1556, 3094, 7327, 16845, 40107, 97132, 202258, 200586, 111118, 54558, 25834, 12372, 5972, 2950, 1374, 756, 386, 239, 112, 67, 60, 38, 97]
- symbol_probs: [6.103515625e-05, 1.6530355424038135e-05, 3.4332275390625e-05, 5.594889444182627e-05, 8.0108642578125e-05, 0.00015894572425168008, 0.00024922689772211015, 0.000415802001953125, 0.0009549458627589047, 0.0019785563927143812, 0.0039342246018350124, 0.0093167619779706, 0.021419525146484375, 0.050998687744140625, 0.1235097274184227, 0.25718435645103455, 0.25505828857421875, 0.14129383862018585, 0.06937408447265625, 0.03284962847828865, 0.0157318115234375, 0.0075937905348837376, 0.0037511189002543688, 0.00174713134765625, 0.0009613037109375, 0.0004908244009129703, 0.00030390420579351485, 0.00014241535973269492, 8.519490802427754e-05, 7.62939453125e-05, 4.831949991057627e-05, 0.00012334187340456992]
- symbol_entropy_bits: 2.898463487625122
- mean_huffman_bits_per_symbol: 2.7228991190592446
- fixed_bits_per_symbol: 5
- mean_residual_abs: 0.021943713839088257
- residual_std: 0.03090117655444586
- clip_ratio: 0.00014368693033854166

## Artifacts

- prior: `outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_4096calib_stage1_lpips002_ft500/static_residual_grid_semantic_position_leftctx_huffman_prior.json`
- summary: `outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_4096calib_stage1_lpips002_ft500/summary.json`
