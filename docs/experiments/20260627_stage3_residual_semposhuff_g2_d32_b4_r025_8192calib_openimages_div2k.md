# 20260627_stage3_residual_semposhuff_g2_d32_b4_r025_8192calib_openimages_div2k

Date: 2026-06-27T13:45:29

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/fit_stage3_residual_huffman_prior.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --image-root /dpl/open-images-v6/train/data --image-root /dpl/div2k --crop-size 256 --max-images 8192 --batch-size 16 --num-workers 8 --detail-downsample-factor 32 --detail-bits 4 --detail-range 0.25 --coding-mode semantic_position_huffman --semantic-group-count 2 --semantic-group-iters 25 --smoothing-count 1 --wandb-mode offline --run-name 20260627_stage3_residual_semposhuff_g2_d32_b4_r025_8192calib_openimages_div2k
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- prior: outputs/stage3_residual_entropy/20260627_stage3_residual_semposhuff_g2_d32_b4_r025_8192calib_openimages_div2k/static_residual_grid_semantic_position_huffman_prior.json
- num_images: 8192
- crop_size: 256
- detail_downsample_factor: 32
- detail_shape: [3, 8, 8]
- detail_bits: 4
- detail_range: 0.25
- coding_mode: semantic_position_huffman
- smoothing_count: 1
- semantic_group_count: 2
- total_symbols: 1572864
- counts: [137, 110, 335, 1118, 4435, 21324, 117712, 606952, 634884, 148704, 29001, 5976, 1391, 416, 177, 192]
- symbol_probs: [8.710225665709004e-05, 6.993611896177754e-05, 0.00021298725914675742, 0.0007108052377589047, 0.0028196971397846937, 0.01355743408203125, 0.0748392716050148, 0.3858896791934967, 0.40364837646484375, 0.09454345703125, 0.018438339233398438, 0.0037994384765625, 0.0008843739633448422, 0.00026448568678461015, 0.0001125335693359375, 0.0001220703125]
- symbol_entropy_bits: 1.932167887687683
- mean_huffman_bits_per_symbol: 1.9403756459554036
- fixed_bits_per_symbol: 4
- mean_residual_abs: 0.021781434043077752
- residual_std: 0.03047513046705603
- clip_ratio: 0.000156402587890625

## Artifacts

- prior: `outputs/stage3_residual_entropy/20260627_stage3_residual_semposhuff_g2_d32_b4_r025_8192calib_openimages_div2k/static_residual_grid_semantic_position_huffman_prior.json`
- summary: `outputs/stage3_residual_entropy/20260627_stage3_residual_semposhuff_g2_d32_b4_r025_8192calib_openimages_div2k/summary.json`
