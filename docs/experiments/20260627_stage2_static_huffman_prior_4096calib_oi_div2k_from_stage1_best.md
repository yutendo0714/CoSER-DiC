# 20260627_stage2_static_huffman_prior_4096calib_oi_div2k_from_stage1_best

Date: 2026-06-27T10:00:03

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/fit_stage2_static_huffman_prior.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --image-root /dpl/open-images-v6/train/data --image-root /dpl/div2k --max-images 4096 --batch-size 16 --num-workers 8 --prior-kind global --smoothing-count 1 --wandb-mode offline --run-name 20260627_stage2_static_huffman_prior_4096calib_oi_div2k_from_stage1_best
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- num_images: 4096
- crop_size: 256
- codebook_size: 8192
- prior_kind: global
- position_backoff_mass: 0.0
- context_topk: 64
- context_backoff_mass: 256.0
- semantic_token_shape: [8, 8]
- total_tokens: 262144
- active_codes: 7520
- global_entropy_bits: 11.919707298278809
- fixed_bits_per_token: 13
- mean_huffman_bits_per_token: 11.956127166748047
- mean_huffman_payload_bpp: 0.011729210615158081
- fixed_bits_payload_bpp: 0.0126953125
- mean_huffman_payload_bytes: 96.085693359375
- payload_bpp_delta_vs_fixed: -0.0009661018848419189

## Artifacts

- prior: `outputs/stage2_semantic_entropy/20260627_stage2_static_huffman_prior_4096calib_oi_div2k_from_stage1_best/static_huffman_prior.json`
- summary: `outputs/stage2_semantic_entropy/20260627_stage2_static_huffman_prior_4096calib_oi_div2k_from_stage1_best/summary.json`
- output_dir: `outputs/stage2_semantic_entropy/20260627_stage2_static_huffman_prior_4096calib_oi_div2k_from_stage1_best`
