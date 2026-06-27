# 20260627_stage3_residual_huffman_d32_b4_r025_4096calib_clicval64_eval

Date: 2026-06-27T11:45:28

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage3_uniform_residual_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --semantic-prior outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json --detail-codec huffman --detail-prior outputs/stage3_residual_entropy/20260627_stage3_residual_huffman_d32_b4_r025_4096calib_oi_div2k/static_residual_grid_huffman_prior.json --image-root /dpl/clic/professional/valid --max-images 64 --batch-size 4 --num-workers 4 --detail-downsample-factor 32 --detail-bits 4 --detail-range 0.25 --wandb-mode offline --run-name 20260627_stage3_residual_huffman_d32_b4_r025_4096calib_clicval64_eval
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- semantic_prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json
- detail_prior: outputs/stage3_residual_entropy/20260627_stage3_residual_huffman_d32_b4_r025_4096calib_oi_div2k/static_residual_grid_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt
- num_images: 41
- crop_size: 256
- semantic_topk: 512
- detail_downsample_factor: 32
- detail_shape: [3, 8, 8]
- detail_bits: 4
- detail_range: 0.25
- detail_codec: huffman
- residual_code: {'codec': 'static_residual_grid_huffman', 'version': 0, 'bits': 4, 'value_range': 0.25, 'payload_codec': 'huffman', 'huffman': {'codec': 'static_huffman', 'version': 0, 'codebook_size': 16, 'code_lengths': [12, 12, 11, 10, 8, 6, 4, 2, 1, 3, 5, 7, 10, 11, 12, 12]}}
- semantic_payload_bpp_mean: 0.009905559260670731
- detail_payload_bpp_mean: 0.005442549542682927
- total_payload_bpp_mean: 0.015348108803353659
- semantic_only_full_stream_bpp_mean: 0.0699730849847561
- stage3_full_stream_bpp_mean: 0.07675840796493902
- semantic_payload_bytes_mean: 81.14634146341463
- detail_payload_bytes_mean: 44.58536585365854
- stage3_stream_bytes_mean: 628.8048780487804
- semantic_only_psnr_mean: 22.46947255948695
- semantic_only_l1_mean: 0.05876295443442536
- semantic_only_ms_ssim_mean: 0.7211727790716218
- stage3_psnr_mean: 23.034366049417635
- stage3_l1_mean: 0.056612195810530244
- stage3_ms_ssim_mean: 0.7236354663604643
- semantic_topk_hit_rate_mean: 0.7164634146341463
- semantic_token_roundtrip_mean: 1.0
- detail_code_roundtrip_mean: 1.0
- residual_grid_abs_mean_mean: 0.018476074039027457
- residual_grid_std_mean: 0.022409117280891757
- residual_grid_clip_ratio_mean: 0.0
- detail_code_entropy_bits_mean: 1.5283508518847024
- stage3_psnr_delta_vs_semantic_only: 0.5648934899306859
- stage3_l1_delta_vs_semantic_only: -0.0021507586238951187
- stage3_ms_ssim_delta_vs_semantic_only: 0.002462687288842469
- all_semantic_tokens_roundtrip: True
- all_detail_codes_roundtrip: True
- roundtrip_failure_count: 0
- roundtrip_failures: []

## Artifacts

- summary: `results/bitstreams/stage3_uniform_residual/20260627_stage3_residual_huffman_d32_b4_r025_4096calib_clicval64_eval/summary.json`
- sample: `results/bitstreams/stage3_uniform_residual/20260627_stage3_residual_huffman_d32_b4_r025_4096calib_clicval64_eval/stage3_uniform_residual_grid.png`
