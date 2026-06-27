# 20260627_stage3_residual_active_b5_smoke2_kodak_compactv3_crc32

Date: 2026-06-27T14:50:22

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage3_uniform_residual_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --semantic-prior outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json --token-prior-checkpoint checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt --image-root /dpl/kodak --crop-size 256 --max-images 2 --batch-size 2 --num-workers 2 --detail-downsample-factor 32 --detail-bits 5 --detail-range 0.25 --detail-codec hybrid_huffman --detail-prior outputs/stage3_residual_entropy/20260627_stage3_residual_hybrid_pos_semposg4_sm0_d32_b5_r025_8192calib_openimages_div2k/static_residual_grid_hybrid_huffman_prior.json --stream-header-codec compact --stream-checksum-codec crc32 --wandb-mode offline --run-name 20260627_stage3_residual_active_b5_smoke2_kodak_compactv3_crc32
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- semantic_prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json
- detail_prior: outputs/stage3_residual_entropy/20260627_stage3_residual_hybrid_pos_semposg4_sm0_d32_b5_r025_8192calib_openimages_div2k/static_residual_grid_hybrid_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt
- num_images: 2
- crop_size: 256
- semantic_topk: 512
- detail_downsample_factor: 32
- detail_shape: [3, 8, 8]
- detail_bits: 5
- detail_range: 0.25
- detail_codec: hybrid_huffman
- stream_header_codec: compact
- stream_checksum_codec: crc32
- residual_code: {'codec': 'static_residual_grid_hybrid_huffman', 'version': 0, 'bits': 5, 'value_range': 0.25, 'payload_codec': 'hybrid_huffman', 'position_code': {'codec': 'static_residual_grid_position_huffman', 'version': 0, 'bits': 5, 'value_range': 0.25, 'payload_codec': 'position_huffman', 'detail_shape': [3, 8, 8], 'num_position_codes': 192, 'min_code_length': 2, 'max_code_length': 13, 'mean_code_length_unweighted': 8.730631510416666}, 'semantic_position_code': {'codec': 'static_residual_grid_semantic_position_huffman', 'version': 0, 'bits': 5, 'value_range': 0.25, 'payload_codec': 'semantic_position_huffman', 'detail_shape': [3, 8, 8], 'semantic_shape': [8, 8], 'group_count': 4, 'num_position_group_codes': 768, 'min_code_length': 1, 'max_code_length': 17, 'mean_code_length_unweighted': 9.809773763020834}}
- semantic_payload_bpp_mean: 0.00982666015625
- detail_payload_bpp_mean: 0.00787353515625
- total_payload_bpp_mean: 0.0177001953125
- semantic_only_full_stream_bpp_mean: 0.01458740234375
- stage3_full_stream_bpp_mean: 0.022705078125
- semantic_payload_bytes_mean: 80.5
- detail_payload_bytes_mean: 64.5
- stage3_stream_bytes_mean: 186.0
- semantic_only_psnr_mean: 21.33194637298584
- semantic_only_l1_mean: 0.06172133795917034
- semantic_only_ms_ssim_mean: 0.6254557073116302
- stage3_psnr_mean: 21.814970016479492
- stage3_l1_mean: 0.05928020738065243
- stage3_ms_ssim_mean: 0.6328615248203278
- semantic_topk_hit_rate_mean: 0.75
- semantic_token_roundtrip_mean: 1.0
- detail_code_roundtrip_mean: 1.0
- residual_grid_abs_mean_mean: 0.01854937057942152
- residual_grid_std_mean: 0.02275838330388069
- residual_grid_clip_ratio_mean: 0.0
- detail_code_entropy_bits_mean: 2.4665966033935547
- detail_hybrid_semantic_mode_mean: 1.0
- stage3_psnr_delta_vs_semantic_only: 0.48302364349365234
- stage3_l1_delta_vs_semantic_only: -0.002441130578517914
- stage3_ms_ssim_delta_vs_semantic_only: 0.00740581750869751
- all_semantic_tokens_roundtrip: True
- all_detail_codes_roundtrip: True
- roundtrip_failure_count: 0
- roundtrip_failures: []

## Artifacts

- summary: `results/bitstreams/stage3_uniform_residual/20260627_stage3_residual_active_b5_smoke2_kodak_compactv3_crc32/summary.json`
- sample: `results/bitstreams/stage3_uniform_residual/20260627_stage3_residual_active_b5_smoke2_kodak_compactv3_crc32/stage3_uniform_residual_grid.png`
