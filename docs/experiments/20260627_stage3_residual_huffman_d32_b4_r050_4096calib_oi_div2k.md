# 20260627_stage3_residual_huffman_d32_b4_r050_4096calib_oi_div2k

Date: 2026-06-27T11:53:25

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/fit_stage3_residual_huffman_prior.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --image-root /dpl/open-images-v6/train/data --image-root /dpl/div2k --max-images 4096 --batch-size 16 --num-workers 8 --detail-downsample-factor 32 --detail-bits 4 --detail-range 0.5 --smoothing-count 1 --wandb-mode offline --run-name 20260627_stage3_residual_huffman_d32_b4_r050_4096calib_oi_div2k
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- prior: outputs/stage3_residual_entropy/20260627_stage3_residual_huffman_d32_b4_r050_4096calib_oi_div2k/static_residual_grid_huffman_prior.json
- num_images: 4096
- crop_size: 256
- detail_downsample_factor: 32
- detail_shape: [3, 8, 8]
- detail_bits: 4
- detail_range: 0.5
- smoothing_count: 1
- total_symbols: 786432
- counts: [0, 7, 20, 20, 77, 800, 12902, 362411, 391434, 17670, 917, 136, 31, 6, 1, 0]
- symbol_probs: [0.0, 8.900959983293433e-06, 2.5431314497836865e-05, 2.5431314497836865e-05, 9.791056072572246e-05, 0.0010172525653615594, 0.0164057407528162, 0.46082940697669983, 0.49773406982421875, 0.02246856689453125, 0.0011660257587209344, 0.00017293293785769492, 3.941853719879873e-05, 7.62939453125e-06, 1.271565793103946e-06, 0.0]
- symbol_entropy_bits: 1.2629764080047607
- mean_huffman_bits_per_symbol: 1.5674578348795574
- fixed_bits_per_symbol: 4
- mean_residual_abs: 0.021804466067502897
- residual_std: 0.030544521252252783
- clip_ratio: 0.0

## Artifacts

- prior: `outputs/stage3_residual_entropy/20260627_stage3_residual_huffman_d32_b4_r050_4096calib_oi_div2k/static_residual_grid_huffman_prior.json`
- summary: `outputs/stage3_residual_entropy/20260627_stage3_residual_huffman_d32_b4_r050_4096calib_oi_div2k/summary.json`
