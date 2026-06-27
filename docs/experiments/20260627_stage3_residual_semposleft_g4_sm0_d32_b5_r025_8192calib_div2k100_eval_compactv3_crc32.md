# 20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_8192calib_div2k100_eval_compactv3_crc32

Date: 2026-06-27T15:24:27

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage3_uniform_residual_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --semantic-prior outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json --token-prior-checkpoint checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt --image-root /dpl/div2k --crop-size 256 --max-images 100 --batch-size 4 --num-workers 4 --detail-downsample-factor 32 --detail-bits 5 --detail-range 0.25 --detail-codec semantic_position_leftctx_huffman --detail-prior outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_8192calib_openimages_div2k/static_residual_grid_semantic_position_leftctx_huffman_prior.json --stream-header-codec compact --stream-checksum-codec crc32 --wandb-mode offline --run-name 20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_8192calib_div2k100_eval_compactv3_crc32
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- semantic_prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json
- detail_prior: outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_8192calib_openimages_div2k/static_residual_grid_semantic_position_leftctx_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt
- num_images: 100
- crop_size: 256
- semantic_topk: 512
- detail_downsample_factor: 32
- detail_shape: [3, 8, 8]
- detail_bits: 5
- detail_range: 0.25
- detail_codec: semantic_position_leftctx_huffman
- stream_header_codec: compact
- stream_checksum_codec: crc32
- main_bpp_metric: actual_payload_bpp_mean
- paper_bpp_metric: paper_bpp_mean
- debug_bpp_metric: debug_full_stream_bpp_mean
- residual_code: {'codec': 'static_residual_grid_semantic_position_leftctx_huffman', 'version': 0, 'bits': 5, 'value_range': 0.25, 'payload_codec': 'semantic_position_leftctx_huffman', 'detail_shape': [3, 8, 8], 'semantic_shape': [8, 8], 'group_count': 4, 'left_context_mode': 'start_left_sign4', 'left_context_count': 4, 'num_position_group_context_codes': 3072, 'min_code_length': 1, 'max_code_length': 18, 'mean_code_length_unweighted': 8.557993570963541}
- semantic_payload_bpp_mean: 0.0104248046875
- detail_payload_bpp_mean: 0.007965087890625
- actual_payload_bpp_mean: 0.018389892578125
- paper_bpp_mean: 0.018389892578125
- total_payload_bpp_mean: 0.018389892578125
- debug_semantic_only_full_stream_bpp_mean: 0.015185546875
- debug_full_stream_bpp_mean: 0.023394775390625
- semantic_only_full_stream_bpp_mean: 0.015185546875
- stage3_full_stream_bpp_mean: 0.023394775390625
- semantic_payload_bytes_mean: 85.4
- detail_payload_bytes_mean: 65.25
- stage3_stream_bytes_mean: 191.65
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

- summary: `results/bitstreams/stage3_uniform_residual/20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_8192calib_div2k100_eval_compactv3_crc32/summary.json`
- sample: `results/bitstreams/stage3_uniform_residual/20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_8192calib_div2k100_eval_compactv3_crc32/stage3_uniform_residual_grid.png`
