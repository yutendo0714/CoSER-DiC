# 20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_4096calib_stage1_rateprior0005_tau01_vq005_ft500

Date: 2026-06-27T22:42:21

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/fit_stage3_residual_huffman_prior.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500.pt --image-root /dpl/open-images-v6/train/data --image-root /dpl/div2k --max-images 4096 --batch-size 16 --num-workers 8 --detail-downsample-factor 32 --detail-bits 4 --detail-range 0.25 --coding-mode semantic_position_leftctx_huffman --semantic-group-count 4 --semantic-group-iters 20 --smoothing-count 0 --wandb-mode offline --run-name 20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_4096calib_stage1_rateprior0005_tau01_vq005_ft500
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500.pt
- prior: outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_4096calib_stage1_rateprior0005_tau01_vq005_ft500/static_residual_grid_semantic_position_leftctx_huffman_prior.json
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
- counts: [63, 53, 178, 567, 2378, 10897, 59355, 296428, 321630, 74838, 15475, 3288, 843, 233, 116, 90]
- symbol_probs: [8.0108642578125e-05, 6.739298260072246e-05, 0.00022633869957644492, 0.000720977783203125, 0.0030237834434956312, 0.01385625172406435, 0.07547378540039062, 0.3769277036190033, 0.40897369384765625, 0.09516143798828125, 0.019677480682730675, 0.004180908203125, 0.001071929931640625, 0.00029627481126226485, 0.00014750163245480508, 0.00011444091796875]
- symbol_entropy_bits: 1.9475233554840088
- mean_huffman_bits_per_symbol: 1.8195559183756511
- fixed_bits_per_symbol: 4
- mean_residual_abs: 0.02196776042789376
- residual_std: 0.030883999876891024
- clip_ratio: 0.00014623006184895834

## Artifacts

- prior: `outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_4096calib_stage1_rateprior0005_tau01_vq005_ft500/static_residual_grid_semantic_position_leftctx_huffman_prior.json`
- summary: `outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_4096calib_stage1_rateprior0005_tau01_vq005_ft500/summary.json`
