# 20260627_stage3_residual_huffman_d32_b4_r025_4096calib_oi_div2k

Date: 2026-06-27T11:43:53

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/fit_stage3_residual_huffman_prior.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --image-root /dpl/open-images-v6/train/data --image-root /dpl/div2k --max-images 4096 --batch-size 16 --num-workers 8 --detail-downsample-factor 32 --detail-bits 4 --detail-range 0.25 --smoothing-count 1 --wandb-mode offline --run-name 20260627_stage3_residual_huffman_d32_b4_r025_4096calib_oi_div2k
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- prior: outputs/stage3_residual_entropy/20260627_stage3_residual_huffman_d32_b4_r025_4096calib_oi_div2k/static_residual_grid_huffman_prior.json
- num_images: 4096
- crop_size: 256
- detail_downsample_factor: 32
- detail_shape: [3, 8, 8]
- detail_bits: 4
- detail_range: 0.25
- smoothing_count: 1
- total_symbols: 786432
- counts: [69, 55, 180, 620, 2248, 10654, 58723, 303688, 317170, 74264, 14629, 3041, 688, 229, 85, 89]
- symbol_probs: [8.7738037109375e-05, 6.993611896177754e-05, 0.0002288818359375, 0.0007883707876317203, 0.0028584797400981188, 0.013547261245548725, 0.07467015832662582, 0.3861592710018158, 0.40330252051353455, 0.0944315567612648, 0.018601736053824425, 0.0038668315391987562, 0.0008748372201807797, 0.00029118856764398515, 0.00010808309161802754, 0.00011316934978822246]
- symbol_entropy_bits: 1.9338687658309937
- mean_huffman_bits_per_symbol: 2.0087242126464844
- fixed_bits_per_symbol: 4
- mean_residual_abs: 0.021804466067502897
- residual_std: 0.030544521252252783
- clip_ratio: 0.00013605753580729166

## Artifacts

- prior: `outputs/stage3_residual_entropy/20260627_stage3_residual_huffman_d32_b4_r025_4096calib_oi_div2k/static_residual_grid_huffman_prior.json`
- summary: `outputs/stage3_residual_entropy/20260627_stage3_residual_huffman_d32_b4_r025_4096calib_oi_div2k/summary.json`
