# 20260627_stage2_position_huffman_prior_512calib_backoff256_from_stage1_best

Date: 2026-06-27T09:45:14

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/fit_stage2_static_huffman_prior.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --max-images 512 --batch-size 8 --num-workers 4 --smoothing-count 0 --prior-kind position --position-backoff-mass 256 --wandb-mode offline --run-name 20260627_stage2_position_huffman_prior_512calib_backoff256_from_stage1_best
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- num_images: 256
- crop_size: 256
- codebook_size: 8192
- prior_kind: position
- position_backoff_mass: 256.0
- semantic_token_shape: [8, 8]
- total_tokens: 16384
- active_codes: 4725
- global_entropy_bits: 11.69194221496582
- fixed_bits_per_token: 13
- mean_huffman_bits_per_token: 8.875244140625
- mean_huffman_payload_bpp: 0.008725643157958984
- fixed_bits_payload_bpp: 0.0126953125
- mean_huffman_payload_bytes: 71.48046875
- payload_bpp_delta_vs_fixed: -0.003969669342041016

## Artifacts

- prior: `outputs/stage2_semantic_entropy/20260627_stage2_position_huffman_prior_512calib_backoff256_from_stage1_best/static_position_huffman_prior.json`
- summary: `outputs/stage2_semantic_entropy/20260627_stage2_position_huffman_prior_512calib_backoff256_from_stage1_best/summary.json`
- output_dir: `outputs/stage2_semantic_entropy/20260627_stage2_position_huffman_prior_512calib_backoff256_from_stage1_best`
