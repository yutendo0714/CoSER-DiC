# 20260627_stage3_residual_hybrid_pos_semposg8_d32_b4_r025_8192calib_div2k100_eval_compactv3_crc32

Date: 2026-06-27T14:05:18

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage3_uniform_residual_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --semantic-prior outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json --token-prior-checkpoint checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt --image-root /dpl/div2k --crop-size 256 --max-images 100 --batch-size 4 --num-workers 4 --detail-downsample-factor 32 --detail-bits 4 --detail-range 0.25 --detail-codec hybrid_huffman --detail-prior outputs/stage3_residual_entropy/20260627_stage3_residual_hybrid_pos_semposg8_d32_b4_r025_8192calib_openimages_div2k/static_residual_grid_hybrid_huffman_prior.json --stream-header-codec compact --stream-checksum-codec crc32 --wandb-mode offline --run-name 20260627_stage3_residual_hybrid_pos_semposg8_d32_b4_r025_8192calib_div2k100_eval_compactv3_crc32
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- semantic_prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json
- detail_prior: outputs/stage3_residual_entropy/20260627_stage3_residual_hybrid_pos_semposg8_d32_b4_r025_8192calib_openimages_div2k/static_residual_grid_hybrid_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt
- num_images: 100
- crop_size: 256
- semantic_topk: 512
- detail_downsample_factor: 32
- detail_shape: [3, 8, 8]
- detail_bits: 4
- detail_range: 0.25
- detail_codec: hybrid_huffman
- stream_header_codec: compact
- stream_checksum_codec: crc32
- residual_code: {'codec': 'static_residual_grid_hybrid_huffman', 'version': 0, 'bits': 4, 'value_range': 0.25, 'payload_codec': 'hybrid_huffman', 'position_code': {'codec': 'static_residual_grid_position_huffman', 'version': 0, 'bits': 4, 'value_range': 0.25, 'payload_codec': 'position_huffman', 'detail_shape': [3, 8, 8], 'num_position_codes': 192, 'min_code_length': 1, 'max_code_length': 14, 'mean_code_length_unweighted': 7.81640625}, 'semantic_position_code': {'codec': 'static_residual_grid_semantic_position_huffman', 'version': 0, 'bits': 4, 'value_range': 0.25, 'payload_codec': 'semantic_position_huffman', 'detail_shape': [3, 8, 8], 'semantic_shape': [8, 8], 'group_count': 8, 'num_position_group_codes': 1536, 'min_code_length': 1, 'max_code_length': 12, 'mean_code_length_unweighted': 6.437540690104167}}
- semantic_payload_bpp_mean: 0.0104248046875
- detail_payload_bpp_mean: 0.00562744140625
- total_payload_bpp_mean: 0.01605224609375
- semantic_only_full_stream_bpp_mean: 0.015185546875
- stage3_full_stream_bpp_mean: 0.02105712890625
- semantic_payload_bytes_mean: 85.4
- detail_payload_bytes_mean: 46.1
- stage3_stream_bytes_mean: 172.5
- semantic_only_psnr_mean: 20.73609987258911
- semantic_only_l1_mean: 0.06987479756120592
- semantic_only_ms_ssim_mean: 0.6836351299285889
- stage3_psnr_mean: 21.041586112976074
- stage3_l1_mean: 0.06801506763324142
- stage3_ms_ssim_mean: 0.6876618662476539
- semantic_topk_hit_rate_mean: 0.64609375
- semantic_token_roundtrip_mean: 1.0
- detail_code_roundtrip_mean: 1.0
- residual_grid_abs_mean_mean: 0.021703385235741734
- residual_grid_std_mean: 0.025807176772505044
- residual_grid_clip_ratio_mean: 0.0
- detail_code_entropy_bits_mean: 1.6949766510725022
- detail_hybrid_semantic_mode_mean: 0.68
- stage3_psnr_delta_vs_semantic_only: 0.30548624038696204
- stage3_l1_delta_vs_semantic_only: -0.0018597299279645013
- stage3_ms_ssim_delta_vs_semantic_only: 0.004026736319065027
- all_semantic_tokens_roundtrip: True
- all_detail_codes_roundtrip: True
- roundtrip_failure_count: 0
- roundtrip_failures: []

## Artifacts

- summary: `results/bitstreams/stage3_uniform_residual/20260627_stage3_residual_hybrid_pos_semposg8_d32_b4_r025_8192calib_div2k100_eval_compactv3_crc32/summary.json`
- sample: `results/bitstreams/stage3_uniform_residual/20260627_stage3_residual_hybrid_pos_semposg8_d32_b4_r025_8192calib_div2k100_eval_compactv3_crc32/stage3_uniform_residual_grid.png`
