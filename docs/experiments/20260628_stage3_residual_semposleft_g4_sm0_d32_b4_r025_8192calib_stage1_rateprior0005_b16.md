# 20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_stage1_rateprior0005_b16

Date: 2026-06-28T00:27:07

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/fit_stage3_residual_huffman_prior.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt --image-root /dpl/open-images-v6/train/data --image-root /dpl/div2k --crop-size 256 --max-images 8192 --batch-size 16 --num-workers 8 --detail-downsample-factor 32 --detail-bits 4 --detail-range 0.25 --coding-mode semantic_position_leftctx_huffman --semantic-group-count 4 --semantic-group-iters 25 --smoothing-count 0 --wandb-mode offline --run-name 20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_stage1_rateprior0005_b16
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt
- prior: outputs/stage3_residual_entropy/20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json
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
- counts: [120, 113, 313, 981, 4136, 19638, 108105, 572738, 667485, 157224, 32599, 6888, 1660, 458, 203, 203]
- symbol_probs: [7.62939453125e-05, 7.184346759459004e-05, 0.00019900004554074258, 0.0006237030029296875, 0.0026295979041606188, 0.012485504150390625, 0.06873130798339844, 0.36413702368736267, 0.4243755340576172, 0.0999603271484375, 0.020725885406136513, 0.0043792724609375, 0.0010553995380178094, 0.00029118856764398515, 0.00012906391930300742, 0.00012906391930300742]
- symbol_entropy_bits: 1.933129072189331
- mean_huffman_bits_per_symbol: 1.813092549641927
- fixed_bits_per_symbol: 4
- mean_residual_abs: 0.02178993674897356
- residual_std: 0.030656913694843572
- clip_ratio: 0.00013987223307291666

## Artifacts

- prior: `outputs/stage3_residual_entropy/20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json`
- summary: `outputs/stage3_residual_entropy/20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_stage1_rateprior0005_b16/summary.json`
