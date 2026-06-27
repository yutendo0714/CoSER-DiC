# 20260627_stage3_b5_semposleft_g4_gencodec_reproduction_perceptual

Date: 2026-06-27T16:37:28

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage3_uniform_residual_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --semantic-prior outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json --token-prior-checkpoint checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt --eval-protocol gencodec_reproduction --crop-size 256 --batch-size 4 --num-workers 4 --detail-downsample-factor 32 --detail-bits 5 --detail-range 0.25 --detail-codec semantic_position_leftctx_huffman --detail-prior outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_8192calib_openimages_div2k/static_residual_grid_semantic_position_leftctx_huffman_prior.json --stream-header-codec compact --stream-checksum-codec crc32 --compute-perceptual --wandb-mode offline --run-name 20260627_stage3_b5_semposleft_g4_gencodec_reproduction_perceptual
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- semantic_prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json
- detail_prior: outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_8192calib_openimages_div2k/static_residual_grid_semantic_position_leftctx_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt
- num_images: 552
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
- eval_protocol: gencodec_reproduction
- eval_datasets: []
- eval_image_roots: ['/dpl/kodak', '/dpl/clic/professional/test', '/dpl/clic/mobile/test', '/dpl/div2k']
- eval_protocol_summary: {'name': 'gencodec_reproduction', 'display_name': 'GenCodec / CoD / CoD-Lite reproduction protocol', 'dataset_keys': ['kodak', 'clic2020_test', 'div2k_val'], 'dataset_counts': {'kodak': 24, 'clic2020_test': 428, 'div2k_val': 100}, 'expected_counts': {'kodak': 24, 'clic2020_test': 428, 'div2k_val': 100}, 'count_status': {'kodak': 'ok', 'clic2020_test': 'ok', 'div2k_val': 'ok'}, 'total_images': 552, 'primary_metrics': ['PSNR', 'LPIPS', 'DISTS', 'FID'], 'bpp_metric': 'actual_payload_bpp for CoSER; official nominal/file bpp must be labeled for external codecs', 'default_crop_size': 512, 'notes': ['Use CLIC2020 test 428 as professional/test plus mobile/test.', 'Use DIV2K validation images 0801-0900, not the first 100 files from a mixed DIV2K root.', 'Patch-based or overlapped-patch FID must be labeled when matching CoD/CoD-Lite.'], 'datasets': [{'key': 'kodak', 'display_name': 'Kodak 24', 'count': 24, 'expected_count': 24, 'status': 'ok', 'source_roots': ['/dpl/kodak'], 'first_path': '/dpl/kodak/kodim01.png', 'last_path': '/dpl/kodak/kodim24.png', 'notes': []}, {'key': 'clic2020_test', 'display_name': 'CLIC2020 test 428', 'count': 428, 'expected_count': 428, 'status': 'ok', 'source_roots': ['/dpl/clic/professional/test', '/dpl/clic/mobile/test'], 'first_path': '/dpl/clic/professional/test/01443654fdc57490d5c2d8b0fd9e065b.png', 'last_path': '/dpl/clic/mobile/test/ff52b7548f7024b96d2cea64e5e061b7.png', 'notes': ['professional/test and mobile/test are concatenated in that order.']}, {'key': 'div2k_val', 'display_name': 'DIV2K validation 100', 'count': 100, 'expected_count': 100, 'status': 'ok', 'source_roots': ['/dpl/div2k'], 'first_path': '/dpl/div2k/0801.png', 'last_path': '/dpl/div2k/0900.png', 'notes': ['Resolved by filename indices 0801-0900 from a mixed DIV2K root.']}]}
- compute_perceptual: True
- residual_code: {'codec': 'static_residual_grid_semantic_position_leftctx_huffman', 'version': 0, 'bits': 5, 'value_range': 0.25, 'payload_codec': 'semantic_position_leftctx_huffman', 'detail_shape': [3, 8, 8], 'semantic_shape': [8, 8], 'group_count': 4, 'left_context_mode': 'start_left_sign4', 'left_context_count': 4, 'num_position_group_context_codes': 3072, 'min_code_length': 1, 'max_code_length': 18, 'mean_code_length_unweighted': 8.557993570963541}
- semantic_payload_bpp_mean: 0.009954259015511776
- detail_payload_bpp_mean: 0.007781761280004529
- actual_payload_bpp_mean: 0.017736020295516304
- paper_bpp_mean: 0.017736020295516304
- total_payload_bpp_mean: 0.017736020295516304
- debug_semantic_only_full_stream_bpp_mean: 0.014715001203011776
- debug_full_stream_bpp_mean: 0.022741787675498188
- semantic_only_full_stream_bpp_mean: 0.014715001203011776
- stage3_full_stream_bpp_mean: 0.022741787675498188
- semantic_payload_bytes_mean: 81.54528985507247
- detail_payload_bytes_mean: 63.7481884057971
- stage3_stream_bytes_mean: 186.30072463768116
- semantic_only_psnr_mean: 21.58319341963616
- semantic_only_l1_mean: 0.061530788434142974
- semantic_only_ms_ssim_mean: 0.7168453826393554
- stage3_psnr_mean: 22.1377178326897
- stage3_l1_mean: 0.05848028325556737
- stage3_ms_ssim_mean: 0.7222157243666225
- semantic_topk_hit_rate_mean: 0.7019078351449275
- semantic_token_roundtrip_mean: 1.0
- detail_code_roundtrip_mean: 1.0
- residual_grid_abs_mean_mean: 0.020801843120379075
- residual_grid_std_mean: 0.02469131398095709
- residual_grid_clip_ratio_mean: 0.0
- detail_code_entropy_bits_mean: 2.4993797757703327
- semantic_only_lpips_alex_mean: 0.6539623773398647
- semantic_only_dists_mean: 0.4057688379417295
- stage3_lpips_alex_mean: 0.6479444390473267
- stage3_dists_mean: 0.40105046968961106
- stage3_psnr_delta_vs_semantic_only: 0.5545244130535387
- stage3_l1_delta_vs_semantic_only: -0.0030505051785756077
- stage3_ms_ssim_delta_vs_semantic_only: 0.005370341727267136
- stage3_lpips_alex_delta_vs_semantic_only: -0.00601793829253805
- stage3_dists_delta_vs_semantic_only: -0.0047183682521184656
- all_semantic_tokens_roundtrip: True
- all_detail_codes_roundtrip: True
- roundtrip_failure_count: 0
- roundtrip_failures: []
- per_image_metrics: results/bitstreams/stage3_uniform_residual/20260627_stage3_b5_semposleft_g4_gencodec_reproduction_perceptual/per_image_metrics.jsonl

## Artifacts

- summary: `results/bitstreams/stage3_uniform_residual/20260627_stage3_b5_semposleft_g4_gencodec_reproduction_perceptual/summary.json`
- per_image_metrics: `results/bitstreams/stage3_uniform_residual/20260627_stage3_b5_semposleft_g4_gencodec_reproduction_perceptual/per_image_metrics.jsonl`
- sample: `results/bitstreams/stage3_uniform_residual/20260627_stage3_b5_semposleft_g4_gencodec_reproduction_perceptual/stage3_uniform_residual_grid.png`
