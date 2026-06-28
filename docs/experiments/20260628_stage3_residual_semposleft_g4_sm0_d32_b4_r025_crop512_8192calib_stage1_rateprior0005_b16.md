# 20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_crop512_8192calib_stage1_rateprior0005_b16

Date: 2026-06-28T02:39:58

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/fit_stage3_residual_huffman_prior.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt --image-root /dpl/open-images-v6/validation/data --crop-size 512 --max-images 8192 --batch-size 16 --num-workers 8 --detail-downsample-factor 32 --detail-bits 4 --detail-range 0.25 --coding-mode semantic_position_leftctx_huffman --semantic-group-count 4 --semantic-group-iters 25 --smoothing-count 0 --wandb-mode offline --run-name 20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_crop512_8192calib_stage1_rateprior0005_b16
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt
- prior: outputs/stage3_residual_entropy/20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_crop512_8192calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json
- num_images: 8192
- crop_size: 512
- detail_downsample_factor: 32
- detail_shape: [3, 16, 16]
- detail_bits: 4
- detail_range: 0.25
- coding_mode: semantic_position_leftctx_huffman
- smoothing_count: 0
- semantic_group_count: 4
- left_context_count: 4
- total_symbols: 6291456
- counts: [364, 393, 1027, 3457, 13257, 62286, 354118, 2285050, 2851677, 573374, 112290, 24301, 6343, 2127, 837, 555]
- symbol_probs: [5.785624307463877e-05, 6.246566772460938e-05, 0.0001632372586755082, 0.0005494753713719547, 0.0021071434020996094, 0.009900093078613281, 0.056285541504621506, 0.3631989061832428, 0.4532618522644043, 0.09113534539937973, 0.01784801483154297, 0.003862540004774928, 0.0010081926593557, 0.0003380775451660156, 0.00013303756713867188, 8.821487426757812e-05]
- symbol_entropy_bits: 1.8425307273864746
- mean_huffman_bits_per_symbol: 1.7258291244506836
- fixed_bits_per_symbol: 4
- mean_residual_abs: 0.02015992786376349
- residual_std: 0.028817793227288786
- clip_ratio: 0.00010172526041666667

## Artifacts

- prior: `outputs/stage3_residual_entropy/20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_crop512_8192calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json`
- summary: `outputs/stage3_residual_entropy/20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_crop512_8192calib_stage1_rateprior0005_b16/summary.json`
