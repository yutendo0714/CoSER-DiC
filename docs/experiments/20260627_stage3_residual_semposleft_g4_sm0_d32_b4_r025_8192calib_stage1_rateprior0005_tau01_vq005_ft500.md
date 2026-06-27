# 20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_stage1_rateprior0005_tau01_vq005_ft500

Date: 2026-06-27T23:10:30

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/fit_stage3_residual_huffman_prior.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500.pt --image-root /dpl/open-images-v6/train/data --image-root /dpl/div2k --max-images 8192 --batch-size 16 --num-workers 8 --detail-downsample-factor 32 --detail-bits 4 --detail-range 0.25 --coding-mode semantic_position_leftctx_huffman --semantic-group-count 4 --semantic-group-iters 20 --smoothing-count 0 --wandb-mode offline --run-name 20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_stage1_rateprior0005_tau01_vq005_ft500
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500.pt
- prior: outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_stage1_rateprior0005_tau01_vq005_ft500/static_residual_grid_semantic_position_leftctx_huffman_prior.json
- num_images: 8192
- crop_size: 256
- detail_downsample_factor: 32
- detail_shape: [3, 8, 8]
- detail_bits: 4
- detail_range: 0.25
- coding_mode: semantic_position_leftctx_huffman
- smoothing_count: 0
- semantic_group_count: 4
- left_context_count: 4
- total_symbols: 1572864
- counts: [122, 115, 310, 1013, 4546, 21703, 117515, 594064, 644626, 149402, 30624, 6391, 1613, 429, 205, 186]
- symbol_probs: [7.756551349302754e-05, 7.311502849915996e-05, 0.00019709269690793008, 0.0006440480356104672, 0.0028902690391987562, 0.013798396103084087, 0.07471402734518051, 0.3776957094669342, 0.4098421633243561, 0.09498723596334457, 0.01947021484375, 0.0040632882155478, 0.0010255178203806281, 0.0002727508544921875, 0.00013033549475949258, 0.000118255615234375]
- symbol_entropy_bits: 1.9406776428222656
- mean_huffman_bits_per_symbol: 1.8182093302408855
- fixed_bits_per_symbol: 4
- mean_residual_abs: 0.021850304794497788
- residual_std: 0.03067176584584372
- clip_ratio: 0.00013796488444010416

## Artifacts

- prior: `outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_stage1_rateprior0005_tau01_vq005_ft500/static_residual_grid_semantic_position_leftctx_huffman_prior.json`
- summary: `outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_stage1_rateprior0005_tau01_vq005_ft500/summary.json`
