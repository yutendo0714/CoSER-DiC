# 20260627_stage3_b5_semposleft_g4_coser_common_lic_gain050_perceptual

Date: 2026-06-27T16:56:56

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage3_uniform_residual_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --semantic-prior outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json --token-prior-checkpoint checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt --eval-protocol coser_common_lic --batch-size 4 --num-workers 4 --detail-downsample-factor 32 --detail-bits 5 --detail-range 0.25 --detail-gain 0.5 --detail-codec semantic_position_leftctx_huffman --detail-prior outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_8192calib_openimages_div2k/static_residual_grid_semantic_position_leftctx_huffman_prior.json --stream-header-codec compact --stream-checksum-codec crc32 --compute-perceptual --run-name 20260627_stage3_b5_semposleft_g4_coser_common_lic_gain050_perceptual
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- semantic_prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json
- detail_prior: outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_8192calib_openimages_div2k/static_residual_grid_semantic_position_leftctx_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt
- num_images: 165
- crop_size: 256
- semantic_topk: 512
- detail_downsample_factor: 32
- detail_shape: [3, 8, 8]
- detail_bits: 5
- detail_range: 0.25
- detail_gain: 0.5
- detail_codec: semantic_position_leftctx_huffman
- stream_header_codec: compact
- stream_checksum_codec: crc32
- main_bpp_metric: actual_payload_bpp_mean
- paper_bpp_metric: paper_bpp_mean
- debug_bpp_metric: debug_full_stream_bpp_mean
- eval_protocol: coser_common_lic
- eval_datasets: []
- eval_image_roots: ['/dpl/kodak', '/dpl/clic/professional/valid', '/dpl/div2k']
- eval_protocol_summary: {'name': 'coser_common_lic', 'display_name': 'CoSER common LIC protocol', 'dataset_keys': ['kodak', 'clic_professional_valid', 'div2k_val'], 'dataset_counts': {'kodak': 24, 'clic_professional_valid': 41, 'div2k_val': 100}, 'expected_counts': {'kodak': 24, 'clic_professional_valid': 41, 'div2k_val': 100}, 'count_status': {'kodak': 'ok', 'clic_professional_valid': 'ok', 'div2k_val': 'ok'}, 'total_images': 165, 'primary_metrics': ['PSNR', 'MS-SSIM', 'LPIPS', 'DISTS', 'FID/KID when sample handling is defined'], 'bpp_metric': 'actual_payload_bpp', 'default_crop_size': None, 'notes': ['Internal CoSER LIC protocol for Kodak 24, CLIC Professional Validation 41, and DIV2K validation 100.', 'Do not merge this table with CLIC2020 test 428 comparisons.'], 'datasets': [{'key': 'kodak', 'display_name': 'Kodak 24', 'count': 24, 'expected_count': 24, 'status': 'ok', 'source_roots': ['/dpl/kodak'], 'first_path': '/dpl/kodak/kodim01.png', 'last_path': '/dpl/kodak/kodim24.png', 'notes': []}, {'key': 'clic_professional_valid', 'display_name': 'CLIC Professional Validation 41', 'count': 41, 'expected_count': 41, 'status': 'ok', 'source_roots': ['/dpl/clic/professional/valid'], 'first_path': '/dpl/clic/professional/valid/alberto-montalesi-176097.png', 'last_path': '/dpl/clic/professional/valid/zugr-108.png', 'notes': []}, {'key': 'div2k_val', 'display_name': 'DIV2K validation 100', 'count': 100, 'expected_count': 100, 'status': 'ok', 'source_roots': ['/dpl/div2k'], 'first_path': '/dpl/div2k/0801.png', 'last_path': '/dpl/div2k/0900.png', 'notes': ['Resolved by filename indices 0801-0900 from a mixed DIV2K root.']}]}
- compute_perceptual: True
- save_reconstructions: False
- save_reconstruction_limit: 0
- save_reconstruction_triptychs: False
- residual_code: {'codec': 'static_residual_grid_semantic_position_leftctx_huffman', 'version': 0, 'bits': 5, 'value_range': 0.25, 'payload_codec': 'semantic_position_leftctx_huffman', 'detail_shape': [3, 8, 8], 'semantic_shape': [8, 8], 'group_count': 4, 'left_context_mode': 'start_left_sign4', 'left_context_count': 4, 'num_position_group_context_codes': 3072, 'min_code_length': 1, 'max_code_length': 18, 'mean_code_length_unweighted': 8.557993570963541}
- semantic_payload_bpp_mean: 0.010515802556818181
- detail_payload_bpp_mean: 0.007851710464015152
- actual_payload_bpp_mean: 0.018367513020833334
- paper_bpp_mean: 0.018367513020833334
- total_payload_bpp_mean: 0.018367513020833334
- debug_semantic_only_full_stream_bpp_mean: 0.015276544744318181
- debug_full_stream_bpp_mean: 0.02630282315340909
- semantic_only_full_stream_bpp_mean: 0.015276544744318181
- stage3_full_stream_bpp_mean: 0.02630282315340909
- semantic_payload_bytes_mean: 86.14545454545454
- detail_payload_bytes_mean: 64.32121212121213
- stage3_stream_bytes_mean: 215.47272727272727
- semantic_only_psnr_mean: 20.918379350142047
- semantic_only_l1_mean: 0.06731460292005177
- semantic_only_ms_ssim_mean: 0.6891150721997926
- stage3_psnr_mean: 21.237604574723676
- stage3_l1_mean: 0.06514744751268264
- stage3_ms_ssim_mean: 0.692368371739532
- semantic_topk_hit_rate_mean: 0.6510416666666666
- semantic_token_roundtrip_mean: 1.0
- detail_code_roundtrip_mean: 1.0
- residual_grid_abs_mean_mean: 0.021032154047861695
- residual_grid_std_mean: 0.02506721400176034
- residual_grid_clip_ratio_mean: 0.0
- detail_code_entropy_bits_mean: 2.5500061530055422
- semantic_only_lpips_alex_mean: 0.7002241195602851
- semantic_only_dists_mean: 0.41472217530915234
- stage3_lpips_alex_mean: 0.6947591229370146
- stage3_dists_mean: 0.412047786062414
- stage3_psnr_delta_vs_semantic_only: 0.31922522458162916
- stage3_l1_delta_vs_semantic_only: -0.0021671554073691285
- stage3_ms_ssim_delta_vs_semantic_only: 0.003253299539739407
- stage3_lpips_alex_delta_vs_semantic_only: -0.0054649966232704905
- stage3_dists_delta_vs_semantic_only: -0.002674389246738351
- all_semantic_tokens_roundtrip: True
- all_detail_codes_roundtrip: True
- roundtrip_failure_count: 0
- roundtrip_failures: []
- per_image_metrics: results/bitstreams/stage3_uniform_residual/20260627_stage3_b5_semposleft_g4_coser_common_lic_gain050_perceptual/per_image_metrics.jsonl

## Artifacts

- summary: `results/bitstreams/stage3_uniform_residual/20260627_stage3_b5_semposleft_g4_coser_common_lic_gain050_perceptual/summary.json`
- per_image_metrics: `results/bitstreams/stage3_uniform_residual/20260627_stage3_b5_semposleft_g4_coser_common_lic_gain050_perceptual/per_image_metrics.jsonl`
- sample: `results/bitstreams/stage3_uniform_residual/20260627_stage3_b5_semposleft_g4_coser_common_lic_gain050_perceptual/stage3_uniform_residual_grid.png`
