# 20260627_stage2_leftctx_huffman_top32_b4096_512calib_from_stage1_best

Date: 2026-06-27T09:57:40

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/fit_stage2_static_huffman_prior.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --max-images 512 --batch-size 8 --num-workers 4 --prior-kind left_context --context-topk 32 --context-backoff-mass 4096 --smoothing-count 0 --wandb-mode offline --run-name 20260627_stage2_leftctx_huffman_top32_b4096_512calib_from_stage1_best
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- num_images: 256
- crop_size: 256
- codebook_size: 8192
- prior_kind: left_context
- position_backoff_mass: 0.0
- context_topk: 32
- context_backoff_mass: 4096.0
- semantic_token_shape: [8, 8]
- total_tokens: 16384
- active_codes: 4725
- global_entropy_bits: 11.69193172454834
- fixed_bits_per_token: 13
- mean_huffman_bits_per_token: 11.5572509765625
- mean_huffman_payload_bpp: 0.01133871078491211
- fixed_bits_payload_bpp: 0.0126953125
- mean_huffman_payload_bytes: 92.88671875
- left_context_top_tokens: 32
- payload_bpp_delta_vs_fixed: -0.0013566017150878906

## Artifacts

- prior: `outputs/stage2_semantic_entropy/20260627_stage2_leftctx_huffman_top32_b4096_512calib_from_stage1_best/static_left_context_huffman_prior.json`
- summary: `outputs/stage2_semantic_entropy/20260627_stage2_leftctx_huffman_top32_b4096_512calib_from_stage1_best/summary.json`
- output_dir: `outputs/stage2_semantic_entropy/20260627_stage2_leftctx_huffman_top32_b4096_512calib_from_stage1_best`
