# 20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_4096calib_stage1_lpips002_ft500

Date: 2026-06-27T18:27:18

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/fit_stage3_residual_huffman_prior.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_active5k_lpips002_fp32loss_ft500.pt --image-root /dpl/open-images-v6/train/data --image-root /dpl/div2k --max-images 4096 --batch-size 16 --num-workers 8 --detail-downsample-factor 32 --detail-bits 4 --detail-range 0.25 --coding-mode semantic_position_leftctx_huffman --semantic-group-count 4 --semantic-group-iters 20 --smoothing-count 0 --wandb-mode offline --run-name 20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_4096calib_stage1_lpips002_ft500
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_active5k_lpips002_fp32loss_ft500.pt
- prior: outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_4096calib_stage1_lpips002_ft500/static_residual_grid_semantic_position_leftctx_huffman_prior.json
- num_images: 4096
- crop_size: 256
- detail_downsample_factor: 32
- detail_shape: [3, 8, 8]
- detail_bits: 4
- detail_range: 0.25
- coding_mode: semantic_position_leftctx_huffman
- smoothing_count: 0
- semantic_group_count: 4
- left_context_count: 4
- total_symbols: 786432
- counts: [57, 58, 160, 462, 1991, 9467, 54411, 303311, 316697, 77651, 16824, 3830, 988, 290, 117, 118]
- symbol_probs: [7.2479248046875e-05, 7.375081622740254e-05, 0.00020345051598269492, 0.00058746337890625, 0.0025316874962300062, 0.01203791331499815, 0.06918716430664062, 0.38567987084388733, 0.4027010500431061, 0.09873834997415543, 0.021392822265625, 0.0048700966872274876, 0.0012563070049509406, 0.00036875405930913985, 0.000148773193359375, 0.00015004475426394492]
- symbol_entropy_bits: 1.940556287765503
- mean_huffman_bits_per_symbol: 1.8156585693359375
- fixed_bits_per_symbol: 4
- mean_residual_abs: 0.021943527409651626
- residual_std: 0.03090095577167782
- clip_ratio: 0.00014368693033854166

## Artifacts

- prior: `outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_4096calib_stage1_lpips002_ft500/static_residual_grid_semantic_position_leftctx_huffman_prior.json`
- summary: `outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_4096calib_stage1_lpips002_ft500/summary.json`
