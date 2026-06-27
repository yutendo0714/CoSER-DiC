# 20260627_stage3_residual_huffman_d32_b5_r025_4096calib_div2k100_eval

Date: 2026-06-27T11:54:32

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage3_uniform_residual_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --semantic-prior outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json --detail-codec huffman --detail-prior outputs/stage3_residual_entropy/20260627_stage3_residual_huffman_d32_b5_r025_4096calib_oi_div2k/static_residual_grid_huffman_prior.json --image-root /dpl/div2k --max-images 100 --batch-size 4 --num-workers 4 --detail-downsample-factor 32 --detail-bits 5 --detail-range 0.25 --wandb-mode offline --run-name 20260627_stage3_residual_huffman_d32_b5_r025_4096calib_div2k100_eval
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- semantic_prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json
- detail_prior: outputs/stage3_residual_entropy/20260627_stage3_residual_huffman_d32_b5_r025_4096calib_oi_div2k/static_residual_grid_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt
- num_images: 100
- crop_size: 256
- semantic_topk: 512
- detail_downsample_factor: 32
- detail_shape: [3, 8, 8]
- detail_bits: 5
- detail_range: 0.25
- detail_codec: huffman
- residual_code: {'codec': 'static_residual_grid_huffman', 'version': 0, 'bits': 5, 'value_range': 0.25, 'payload_codec': 'huffman', 'huffman': {'codec': 'static_huffman', 'version': 0, 'codebook_size': 32, 'code_lengths': [14, 15, 15, 14, 13, 12, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 14, 13]}}
- semantic_payload_bpp_mean: 0.0104248046875
- detail_payload_bpp_mean: 0.008558349609375
- total_payload_bpp_mean: 0.018983154296875
- semantic_only_full_stream_bpp_mean: 0.070504150390625
- stage3_full_stream_bpp_mean: 0.08041015625
- semantic_payload_bytes_mean: 85.4
- detail_payload_bytes_mean: 70.11
- stage3_stream_bytes_mean: 658.72
- semantic_only_psnr_mean: 20.73609987258911
- semantic_only_l1_mean: 0.06987479756120592
- semantic_only_ms_ssim_mean: 0.6836351299285889
- stage3_psnr_mean: 21.19623002052307
- stage3_l1_mean: 0.06742917627096176
- stage3_ms_ssim_mean: 0.6890050381422043
- semantic_topk_hit_rate_mean: 0.64609375
- semantic_token_roundtrip_mean: 1.0
- detail_code_roundtrip_mean: 1.0
- residual_grid_abs_mean_mean: 0.021703385235741734
- residual_grid_std_mean: 0.025807176772505044
- residual_grid_clip_ratio_mean: 0.0
- detail_code_entropy_bits_mean: 2.5721683251857757
- stage3_psnr_delta_vs_semantic_only: 0.4601301479339597
- stage3_l1_delta_vs_semantic_only: -0.002445621290244157
- stage3_ms_ssim_delta_vs_semantic_only: 0.005369908213615382
- all_semantic_tokens_roundtrip: True
- all_detail_codes_roundtrip: True
- roundtrip_failure_count: 0
- roundtrip_failures: []

## Artifacts

- summary: `results/bitstreams/stage3_uniform_residual/20260627_stage3_residual_huffman_d32_b5_r025_4096calib_div2k100_eval/summary.json`
- sample: `results/bitstreams/stage3_uniform_residual/20260627_stage3_residual_huffman_d32_b5_r025_4096calib_div2k100_eval/stage3_uniform_residual_grid.png`
