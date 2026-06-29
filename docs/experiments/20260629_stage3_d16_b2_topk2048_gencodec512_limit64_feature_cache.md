# 20260629_stage3_d16_b2_topk2048_gencodec512_limit64_feature_cache

Date: 2026-06-29T23:15:41

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage3_uniform_residual_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt --semantic-prior outputs/stage2_learned_entropy/20260628_stage2_learned_topk2048_escape_huffman_fit_32768tokens_crop512_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json --token-prior-checkpoint checkpoints/stage2_token_prior/20260628_stage2_token_prior_crop512_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_amp_es.pt --eval-protocol gencodec_reproduction --allow-protocol-count-mismatch --crop-size 512 --max-images 64 --batch-size 4 --num-workers 4 --detail-downsample-factor 16 --detail-bits 2 --detail-range 0.25 --detail-codec semantic_position_leftctx_huffman --detail-prior outputs/stage3_residual_entropy/20260629_stage3_residual_semposleft_g4_sm0_d16_b2_r025_crop512_2048calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json --decoder-postprocess unsharp3x3 --decoder-postprocess-strength 0.2 --stream-header-codec compact --stream-checksum-codec crc32 --save-reconstructions --save-decoder-feature-cache --compute-perceptual --run-name 20260629_stage3_d16_b2_topk2048_gencodec512_limit64_feature_cache
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt
- semantic_prior: outputs/stage2_learned_entropy/20260628_stage2_learned_topk2048_escape_huffman_fit_32768tokens_crop512_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json
- detail_prior: outputs/stage3_residual_entropy/20260629_stage3_residual_semposleft_g4_sm0_d16_b2_r025_crop512_2048calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260628_stage2_token_prior_crop512_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_amp_es.pt
- num_images: 64
- crop_size: 512
- protocol_default_crop_size: 512
- crop_size_matches_protocol_default: True
- semantic_topk: 2048
- semantic_topk_schedule: prefix_replay_decoder_safe
- detail_downsample_factor: 16
- detail_shape: [3, 32, 32]
- detail_bits: 2
- detail_range: 0.25
- detail_gain: 1.0
- decoder_postprocess: unsharp3x3
- decoder_postprocess_strength: 0.2
- decoder_refiner_checkpoint: 
- decoder_refiner_enabled: False
- decoder_refiner_config: {}
- decoder_refiner_payload_policy: 
- detail_codec: semantic_position_leftctx_huffman
- stream_header_codec: compact
- stream_checksum_codec: crc32
- main_bpp_metric: actual_payload_bpp_mean
- paper_bpp_metric: paper_bpp_mean
- debug_bpp_metric: debug_full_stream_bpp_mean
- eval_protocol: gencodec_reproduction
- eval_datasets: []
- eval_image_roots: ['/dpl/kodak', '/dpl/clic/professional/test', '/dpl/clic/mobile/test', '/dpl/div2k']
- eval_protocol_summary: {'name': 'gencodec_reproduction', 'display_name': 'Legacy alias for CoD 512 reproduction protocol', 'dataset_keys': ['kodak', 'clic2020_test', 'div2k_val'], 'dataset_counts': {'kodak': 24, 'clic2020_test': 428, 'div2k_val': 100}, 'expected_counts': {'kodak': 24, 'clic2020_test': 428, 'div2k_val': 100}, 'count_status': {'kodak': 'ok', 'clic2020_test': 'ok', 'div2k_val': 'ok'}, 'total_images': 552, 'primary_metrics': ['PSNR', 'LPIPS', 'DISTS', 'FID'], 'bpp_metric': 'actual_payload_bpp for CoSER; official nominal/file bpp must be labeled for external codecs', 'default_crop_size': 512, 'notes': ['Legacy name retained for old experiment commands. Prefer cod_reproduction_512 for new runs.', 'Same dataset and 512x512 center-crop preprocessing as cod_reproduction_512.', 'Do not use this name for CoD-Lite full-resolution comparison tables.'], 'datasets': [{'key': 'kodak', 'display_name': 'Kodak 24', 'count': 24, 'expected_count': 24, 'status': 'ok', 'source_roots': ['/dpl/kodak'], 'first_path': '/dpl/kodak/kodim01.png', 'last_path': '/dpl/kodak/kodim24.png', 'notes': []}, {'key': 'clic2020_test', 'display_name': 'CLIC2020 test 428', 'count': 428, 'expected_count': 428, 'status': 'ok', 'source_roots': ['/dpl/clic/professional/test', '/dpl/clic/mobile/test'], 'first_path': '/dpl/clic/professional/test/01443654fdc57490d5c2d8b0fd9e065b.png', 'last_path': '/dpl/clic/mobile/test/ff52b7548f7024b96d2cea64e5e061b7.png', 'notes': ['professional/test and mobile/test are concatenated in that order.']}, {'key': 'div2k_val', 'display_name': 'DIV2K validation 100', 'count': 100, 'expected_count': 100, 'status': 'ok', 'source_roots': ['/dpl/div2k'], 'first_path': '/dpl/div2k/0801.png', 'last_path': '/dpl/div2k/0900.png', 'notes': ['Resolved by filename indices 0801-0900 from a mixed DIV2K root.']}]}
- deterministic_eval: True
- shuffle_images: False
- shuffle_seed: 1234
- compute_perceptual: True
- save_reconstructions: True
- save_reconstruction_limit: 0
- save_reconstruction_triptychs: False
- save_decoder_feature_cache: True
- decoder_feature_cache_dir: results/bitstreams/stage3_uniform_residual/20260629_stage3_d16_b2_topk2048_gencodec512_limit64_feature_cache/decoder_feature_cache
- residual_code: {'codec': 'static_residual_grid_semantic_position_leftctx_huffman', 'version': 0, 'bits': 2, 'value_range': 0.25, 'payload_codec': 'semantic_position_leftctx_huffman', 'detail_shape': [3, 32, 32], 'semantic_shape': [16, 16], 'group_count': 4, 'left_context_mode': 'start_left_sign4', 'left_context_count': 4, 'num_position_group_context_codes': 49152, 'min_code_length': 1, 'max_code_length': 3, 'mean_code_length_unweighted': 2.1702117919921875}
- semantic_payload_bpp_mean: 0.008796215057373047
- detail_payload_bpp_mean: 0.015334606170654297
- actual_payload_bpp_mean: 0.024130821228027344
- paper_bpp_mean: 0.024130821228027344
- total_payload_bpp_mean: 0.024130821228027344
- debug_semantic_only_full_stream_bpp_mean: 0.010016441345214844
- debug_full_stream_bpp_mean: 0.02651071548461914
- semantic_only_full_stream_bpp_mean: 0.010016441345214844
- stage3_full_stream_bpp_mean: 0.02651071548461914
- semantic_payload_bytes_mean: 288.234375
- detail_payload_bytes_mean: 502.484375
- stage3_stream_bytes_mean: 868.703125
- semantic_only_psnr_mean: 21.898068889975548
- semantic_only_l1_mean: 0.05379394179908559
- semantic_only_ms_ssim_mean: 0.7524227481335402
- stage3_psnr_mean: 20.877180874347687
- stage3_l1_mean: 0.06972369481809437
- stage3_ms_ssim_mean: 0.7001581192016602
- semantic_topk_hit_rate_mean: 0.93487548828125
- semantic_token_roundtrip_mean: 1.0
- detail_code_roundtrip_mean: 1.0
- residual_grid_abs_mean_mean: 0.025507103135169018
- residual_grid_std_mean: 0.035717408143682405
- residual_grid_clip_ratio_mean: 0.0006306966374722833
- detail_code_entropy_bits_mean: 1.0038135247305036
- semantic_only_lpips_alex_mean: 0.5489634621771984
- semantic_only_dists_mean: 0.35482229106128216
- stage3_lpips_alex_mean: 0.6951764142140746
- stage3_dists_mean: 0.3708472680300474
- stage3_psnr_delta_vs_semantic_only: -1.020888015627861
- stage3_l1_delta_vs_semantic_only: 0.015929753019008785
- stage3_ms_ssim_delta_vs_semantic_only: -0.05226462893188
- stage3_lpips_alex_delta_vs_semantic_only: 0.14621295203687623
- stage3_dists_delta_vs_semantic_only: 0.01602497696876526
- all_semantic_tokens_roundtrip: True
- all_detail_codes_roundtrip: True
- roundtrip_failure_count: 0
- roundtrip_failures: []
- per_image_metrics: results/bitstreams/stage3_uniform_residual/20260629_stage3_d16_b2_topk2048_gencodec512_limit64_feature_cache/per_image_metrics.jsonl
- reconstruction_manifest: results/bitstreams/stage3_uniform_residual/20260629_stage3_d16_b2_topk2048_gencodec512_limit64_feature_cache/reconstructions/manifest.jsonl
- reconstruction_count: 64
- reconstruction_dirs: {'reference': 'results/bitstreams/stage3_uniform_residual/20260629_stage3_d16_b2_topk2048_gencodec512_limit64_feature_cache/reconstructions/reference', 'semantic_only': 'results/bitstreams/stage3_uniform_residual/20260629_stage3_d16_b2_topk2048_gencodec512_limit64_feature_cache/reconstructions/semantic_only', 'stage3': 'results/bitstreams/stage3_uniform_residual/20260629_stage3_d16_b2_topk2048_gencodec512_limit64_feature_cache/reconstructions/stage3'}

## Artifacts

- summary: `results/bitstreams/stage3_uniform_residual/20260629_stage3_d16_b2_topk2048_gencodec512_limit64_feature_cache/summary.json`
- per_image_metrics: `results/bitstreams/stage3_uniform_residual/20260629_stage3_d16_b2_topk2048_gencodec512_limit64_feature_cache/per_image_metrics.jsonl`
- sample: `results/bitstreams/stage3_uniform_residual/20260629_stage3_d16_b2_topk2048_gencodec512_limit64_feature_cache/stage3_uniform_residual_grid.png`
- reconstruction_manifest: `results/bitstreams/stage3_uniform_residual/20260629_stage3_d16_b2_topk2048_gencodec512_limit64_feature_cache/reconstructions/manifest.jsonl`
