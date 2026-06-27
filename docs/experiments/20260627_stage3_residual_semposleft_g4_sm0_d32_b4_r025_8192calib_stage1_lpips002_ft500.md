# 20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_stage1_lpips002_ft500

Date: 2026-06-27T19:16:29

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/fit_stage3_residual_huffman_prior.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_active5k_lpips002_fp32loss_ft500.pt --image-root /dpl/open-images-v6/train/data --image-root /dpl/div2k --max-images 8192 --batch-size 16 --num-workers 8 --detail-downsample-factor 32 --detail-bits 4 --detail-range 0.25 --coding-mode semantic_position_leftctx_huffman --semantic-group-count 4 --semantic-group-iters 20 --smoothing-count 0 --wandb-mode offline --run-name 20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_stage1_lpips002_ft500
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_active5k_lpips002_fp32loss_ft500.pt
- prior: outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_stage1_lpips002_ft500/static_residual_grid_semantic_position_leftctx_huffman_prior.json
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
- counts: [107, 122, 269, 834, 3856, 18929, 108569, 608436, 632252, 155964, 33301, 7377, 1888, 521, 210, 229]
- symbol_probs: [6.802877032896504e-05, 7.756551349302754e-05, 0.00017102558922488242, 0.000530242919921875, 0.0024515788536518812, 0.012034733779728413, 0.06902631372213364, 0.38683319091796875, 0.40197500586509705, 0.09915924072265625, 0.021172204986214638, 0.0046901702880859375, 0.0012003580341115594, 0.00033124288893304765, 0.000133514404296875, 0.00014559428382199258]
- symbol_entropy_bits: 1.936391830444336
- mean_huffman_bits_per_symbol: 1.8154576619466145
- fixed_bits_per_symbol: 4
- mean_residual_abs: 0.02187740853454064
- residual_std: 0.030731071305588522
- clip_ratio: 0.00013287862141927084

## Artifacts

- prior: `outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_stage1_lpips002_ft500/static_residual_grid_semantic_position_leftctx_huffman_prior.json`
- summary: `outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_stage1_lpips002_ft500/summary.json`
