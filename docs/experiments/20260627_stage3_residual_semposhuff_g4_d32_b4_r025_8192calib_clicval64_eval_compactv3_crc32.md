# 20260627_stage3_residual_semposhuff_g4_d32_b4_r025_8192calib_clicval64_eval_compactv3_crc32

Date: 2026-06-27T13:48:22

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage3_uniform_residual_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --semantic-prior outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json --token-prior-checkpoint checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt --image-root /dpl/clic/professional/valid --image-root /dpl/clic/mobile/valid --crop-size 256 --max-images 64 --batch-size 4 --num-workers 4 --detail-downsample-factor 32 --detail-bits 4 --detail-range 0.25 --detail-codec semantic_position_huffman --detail-prior outputs/stage3_residual_entropy/20260627_stage3_residual_semposhuff_g4_d32_b4_r025_8192calib_openimages_div2k/static_residual_grid_semantic_position_huffman_prior.json --stream-header-codec compact --stream-checksum-codec crc32 --wandb-mode offline --run-name 20260627_stage3_residual_semposhuff_g4_d32_b4_r025_8192calib_clicval64_eval_compactv3_crc32
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- semantic_prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json
- detail_prior: outputs/stage3_residual_entropy/20260627_stage3_residual_semposhuff_g4_d32_b4_r025_8192calib_openimages_div2k/static_residual_grid_semantic_position_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt
- num_images: 64
- crop_size: 256
- semantic_topk: 512
- detail_downsample_factor: 32
- detail_shape: [3, 8, 8]
- detail_bits: 4
- detail_range: 0.25
- detail_codec: semantic_position_huffman
- stream_header_codec: compact
- stream_checksum_codec: crc32
- residual_code: {'codec': 'static_residual_grid_semantic_position_huffman', 'version': 0, 'bits': 4, 'value_range': 0.25, 'payload_codec': 'semantic_position_huffman', 'detail_shape': [3, 8, 8], 'semantic_shape': [8, 8], 'group_count': 4, 'num_position_group_codes': 768, 'min_code_length': 1, 'max_code_length': 12, 'mean_code_length_unweighted': 7.248697916666667}
- semantic_payload_bpp_mean: 0.010049819946289062
- detail_payload_bpp_mean: 0.0052547454833984375
- total_payload_bpp_mean: 0.0153045654296875
- semantic_only_full_stream_bpp_mean: 0.014810562133789062
- stage3_full_stream_bpp_mean: 0.0203094482421875
- semantic_payload_bytes_mean: 82.328125
- detail_payload_bytes_mean: 43.046875
- stage3_stream_bytes_mean: 166.375
- semantic_only_psnr_mean: 22.364622324705124
- semantic_only_l1_mean: 0.05892299117112998
- semantic_only_ms_ssim_mean: 0.7271101297810674
- stage3_psnr_mean: 22.908445432782173
- stage3_l1_mean: 0.05666106294665951
- stage3_ms_ssim_mean: 0.729686098638922
- semantic_topk_hit_rate_mean: 0.6943359375
- semantic_token_roundtrip_mean: 1.0
- detail_code_roundtrip_mean: 1.0
- residual_grid_abs_mean_mean: 0.018713655663304962
- residual_grid_std_mean: 0.022896580419910606
- residual_grid_clip_ratio_mean: 0.0
- detail_code_entropy_bits_mean: 1.5519282510504127
- stage3_psnr_delta_vs_semantic_only: 0.5438231080770493
- stage3_l1_delta_vs_semantic_only: -0.0022619282244704664
- stage3_ms_ssim_delta_vs_semantic_only: 0.0025759688578546047
- all_semantic_tokens_roundtrip: True
- all_detail_codes_roundtrip: True
- roundtrip_failure_count: 0
- roundtrip_failures: []

## Artifacts

- summary: `results/bitstreams/stage3_uniform_residual/20260627_stage3_residual_semposhuff_g4_d32_b4_r025_8192calib_clicval64_eval_compactv3_crc32/summary.json`
- sample: `results/bitstreams/stage3_uniform_residual/20260627_stage3_residual_semposhuff_g4_d32_b4_r025_8192calib_clicval64_eval_compactv3_crc32/stage3_uniform_residual_grid.png`
