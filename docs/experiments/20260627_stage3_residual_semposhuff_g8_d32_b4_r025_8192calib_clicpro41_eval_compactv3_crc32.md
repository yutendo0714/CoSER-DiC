# 20260627_stage3_residual_semposhuff_g8_d32_b4_r025_8192calib_clicpro41_eval_compactv3_crc32

Date: 2026-06-27T13:42:20

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage3_uniform_residual_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --semantic-prior outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json --token-prior-checkpoint checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt --image-root /dpl/clic/professional/valid --crop-size 256 --max-images 41 --batch-size 4 --num-workers 4 --detail-downsample-factor 32 --detail-bits 4 --detail-range 0.25 --detail-codec semantic_position_huffman --detail-prior outputs/stage3_residual_entropy/20260627_stage3_residual_semposhuff_g8_d32_b4_r025_8192calib_openimages_div2k/static_residual_grid_semantic_position_huffman_prior.json --stream-header-codec compact --stream-checksum-codec crc32 --wandb-mode offline --run-name 20260627_stage3_residual_semposhuff_g8_d32_b4_r025_8192calib_clicpro41_eval_compactv3_crc32
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- semantic_prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json
- detail_prior: outputs/stage3_residual_entropy/20260627_stage3_residual_semposhuff_g8_d32_b4_r025_8192calib_openimages_div2k/static_residual_grid_semantic_position_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt
- num_images: 41
- crop_size: 256
- semantic_topk: 512
- detail_downsample_factor: 32
- detail_shape: [3, 8, 8]
- detail_bits: 4
- detail_range: 0.25
- detail_codec: semantic_position_huffman
- stream_header_codec: compact
- stream_checksum_codec: crc32
- residual_code: {'codec': 'static_residual_grid_semantic_position_huffman', 'version': 0, 'bits': 4, 'value_range': 0.25, 'payload_codec': 'semantic_position_huffman', 'detail_shape': [3, 8, 8], 'semantic_shape': [8, 8], 'group_count': 8, 'num_position_group_codes': 1536, 'min_code_length': 1, 'max_code_length': 12, 'mean_code_length_unweighted': 6.437540690104167}
- semantic_payload_bpp_mean: 0.009908536585365854
- detail_payload_bpp_mean: 0.005204363567073171
- total_payload_bpp_mean: 0.015112900152439025
- semantic_only_full_stream_bpp_mean: 0.014669278772865854
- stage3_full_stream_bpp_mean: 0.020117782964939025
- semantic_payload_bytes_mean: 81.17073170731707
- detail_payload_bytes_mean: 42.63414634146341
- stage3_stream_bytes_mean: 164.8048780487805
- semantic_only_psnr_mean: 22.469923763740354
- semantic_only_l1_mean: 0.05875815989495051
- semantic_only_ms_ssim_mean: 0.7211975629736738
- stage3_psnr_mean: 23.03450528586783
- stage3_l1_mean: 0.05661088568953479
- stage3_ms_ssim_mean: 0.7236553219760337
- semantic_topk_hit_rate_mean: 0.7164634146341463
- semantic_token_roundtrip_mean: 1.0
- detail_code_roundtrip_mean: 1.0
- residual_grid_abs_mean_mean: 0.018468380587675224
- residual_grid_std_mean: 0.0224008904315713
- residual_grid_clip_ratio_mean: 0.0
- detail_code_entropy_bits_mean: 1.527035231997327
- stage3_psnr_delta_vs_semantic_only: 0.5645815221274759
- stage3_l1_delta_vs_semantic_only: -0.002147274205415717
- stage3_ms_ssim_delta_vs_semantic_only: 0.002457759002359894
- all_semantic_tokens_roundtrip: True
- all_detail_codes_roundtrip: True
- roundtrip_failure_count: 0
- roundtrip_failures: []

## Artifacts

- summary: `results/bitstreams/stage3_uniform_residual/20260627_stage3_residual_semposhuff_g8_d32_b4_r025_8192calib_clicpro41_eval_compactv3_crc32/summary.json`
- sample: `results/bitstreams/stage3_uniform_residual/20260627_stage3_residual_semposhuff_g8_d32_b4_r025_8192calib_clicpro41_eval_compactv3_crc32/stage3_uniform_residual_grid.png`
