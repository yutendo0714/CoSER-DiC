# 20260627_stage3_residual_huffman_d32_b4_r025_smoke

Date: 2026-06-27T11:43:14

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/fit_stage3_residual_huffman_prior.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --image-root /dpl/open-images-v6/train/data --image-root /dpl/div2k --max-images 16 --batch-size 4 --num-workers 2 --detail-downsample-factor 32 --detail-bits 4 --detail-range 0.25 --smoothing-count 1 --wandb-mode offline --run-name 20260627_stage3_residual_huffman_d32_b4_r025_smoke
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- prior: outputs/stage3_residual_entropy/20260627_stage3_residual_huffman_d32_b4_r025_smoke/static_residual_grid_huffman_prior.json
- num_images: 16
- crop_size: 256
- detail_downsample_factor: 32
- detail_shape: [3, 8, 8]
- detail_bits: 4
- detail_range: 0.25
- smoothing_count: 1
- total_symbols: 3072
- counts: [0, 0, 2, 5, 16, 37, 217, 1260, 1187, 272, 66, 10, 0, 0, 0, 0]
- symbol_probs: [0.0, 0.0, 0.0006510416860692203, 0.0016276041278615594, 0.0052083334885537624, 0.012044270522892475, 0.0706380233168602, 0.41015625, 0.3863932192325592, 0.0885416641831398, 0.021484375, 0.0032552082557231188, 0.0, 0.0, 0.0, 0.0]
- symbol_entropy_bits: 1.921384334564209
- mean_huffman_bits_per_symbol: 1.9951171875
- fixed_bits_per_symbol: 4
- mean_residual_abs: 0.021693953933815163
- residual_std: 0.030292980554100013
- clip_ratio: 0.0

## Artifacts

- prior: `outputs/stage3_residual_entropy/20260627_stage3_residual_huffman_d32_b4_r025_smoke/static_residual_grid_huffman_prior.json`
- summary: `outputs/stage3_residual_entropy/20260627_stage3_residual_huffman_d32_b4_r025_smoke/summary.json`
