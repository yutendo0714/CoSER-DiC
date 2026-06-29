# 20260629_stage3_residual_semposleft_g4_sm0_d16_b2_r025_crop512_2048calib_stage1_rateprior0005_b16

Date: 2026-06-29T23:10:50

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/fit_stage3_residual_huffman_prior.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt --image-root /dpl/open-images-v6/train/data --crop-size 512 --max-images 2048 --batch-size 16 --num-workers 8 --detail-downsample-factor 16 --detail-bits 2 --detail-range 0.25 --coding-mode semantic_position_leftctx_huffman --semantic-group-count 4 --smoothing-count 0 --run-name 20260629_stage3_residual_semposleft_g4_sm0_d16_b2_r025_crop512_2048calib_stage1_rateprior0005_b16 --wandb-mode offline
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt
- prior: outputs/stage3_residual_entropy/20260629_stage3_residual_semposleft_g4_sm0_d16_b2_r025_crop512_2048calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json
- num_images: 2048
- crop_size: 512
- detail_downsample_factor: 16
- detail_shape: [3, 32, 32]
- detail_bits: 2
- detail_range: 0.25
- coding_mode: semantic_position_leftctx_huffman
- smoothing_count: 0
- semantic_group_count: 4
- left_context_count: 4
- total_symbols: 6291456
- counts: [14920, 2813354, 3439299, 23883]
- symbol_probs: [0.0023714702110737562, 0.44717058539390564, 0.5466618537902832, 0.003796100616455078]
- symbol_entropy_bits: 1.0467108488082886
- mean_huffman_bits_per_symbol: 1.3248090744018555
- fixed_bits_per_symbol: 2
- mean_residual_abs: 0.028760460011350613
- residual_std: 0.04255055573004293
- clip_ratio: 0.0008115768432617188

## Artifacts

- prior: `outputs/stage3_residual_entropy/20260629_stage3_residual_semposleft_g4_sm0_d16_b2_r025_crop512_2048calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json`
- summary: `outputs/stage3_residual_entropy/20260629_stage3_residual_semposleft_g4_sm0_d16_b2_r025_crop512_2048calib_stage1_rateprior0005_b16/summary.json`
