# 20260627_stage2_position_huffman_prior_512calib_from_stage1_best

Date: 2026-06-27T09:42:47

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/fit_stage2_static_huffman_prior.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --max-images 512 --batch-size 8 --num-workers 4 --smoothing-count 1 --prior-kind position --wandb-mode offline --run-name 20260627_stage2_position_huffman_prior_512calib_from_stage1_best
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- num_images: 256
- crop_size: 256
- codebook_size: 8192
- prior_kind: position
- semantic_token_shape: [8, 8]
- total_tokens: 16384
- active_codes: 4725
- global_entropy_bits: 11.69193172454834
- fixed_bits_per_token: 13
- mean_huffman_bits_per_token: 12.9151611328125
- mean_huffman_payload_bpp: 0.012667655944824219
- fixed_bits_payload_bpp: 0.0126953125
- mean_huffman_payload_bytes: 103.7734375
- payload_bpp_delta_vs_fixed: -2.765655517578125e-05

## Artifacts

- prior: `outputs/stage2_semantic_entropy/20260627_stage2_position_huffman_prior_512calib_from_stage1_best/static_position_huffman_prior.json`
- summary: `outputs/stage2_semantic_entropy/20260627_stage2_position_huffman_prior_512calib_from_stage1_best/summary.json`
- output_dir: `outputs/stage2_semantic_entropy/20260627_stage2_position_huffman_prior_512calib_from_stage1_best`
