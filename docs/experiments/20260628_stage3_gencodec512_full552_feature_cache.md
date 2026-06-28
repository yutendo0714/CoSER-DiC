# 20260628_stage3_gencodec512_full552_feature_cache

Date: 2026-06-28T18:34:41

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage3_uniform_residual_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt --semantic-prior outputs/stage2_learned_entropy/20260628_stage2_learned_topk2048_escape_huffman_fit_32768tokens_crop512_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json --token-prior-checkpoint checkpoints/stage2_token_prior/20260628_stage2_token_prior_crop512_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_amp_es.pt --detail-prior outputs/stage3_residual_entropy/20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_crop512_8192calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json --eval-protocol gencodec_reproduction --crop-size 512 --batch-size 4 --num-workers 4 --detail-downsample-factor 32 --detail-bits 4 --detail-range 0.25 --detail-gain 1.0 --detail-codec semantic_position_leftctx_huffman --stream-header-codec json --stream-checksum-codec sha256 --save-reconstructions --save-decoder-feature-cache --output-dir results/bitstreams/stage3_uniform_residual --run-name 20260628_stage3_gencodec512_full552_feature_cache --wandb-mode offline
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt
- semantic_prior: outputs/stage2_learned_entropy/20260628_stage2_learned_topk2048_escape_huffman_fit_32768tokens_crop512_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json
- detail_prior: outputs/stage3_residual_entropy/20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_crop512_8192calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260628_stage2_token_prior_crop512_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_amp_es.pt
- num_images: 552
- crop_size: 512
- protocol_default_crop_size: 512
- crop_size_matches_protocol_default: True
- semantic_topk: 2048
- semantic_topk_schedule: prefix_replay_decoder_safe
- detail_downsample_factor: 32
- detail_shape: [3, 16, 16]
- detail_bits: 4
- detail_range: 0.25
- detail_gain: 1.0
- decoder_postprocess: none
- decoder_postprocess_strength: 0.0
- decoder_refiner_checkpoint: 
- decoder_refiner_enabled: False
- decoder_refiner_config: {}
- decoder_refiner_payload_policy: 
- detail_codec: semantic_position_leftctx_huffman
- stream_header_codec: json
- stream_checksum_codec: sha256
- main_bpp_metric: actual_payload_bpp_mean
- paper_bpp_metric: paper_bpp_mean
- debug_bpp_metric: debug_full_stream_bpp_mean
- eval_protocol: gencodec_reproduction
- eval_datasets: []
- eval_image_roots: ['/dpl/kodak', '/dpl/clic/professional/test', '/dpl/clic/mobile/test', '/dpl/div2k']
- eval_protocol_summary: {'name': 'gencodec_reproduction', 'display_name': 'Legacy alias for CoD 512 reproduction protocol', 'dataset_keys': ['kodak', 'clic2020_test', 'div2k_val'], 'dataset_counts': {'kodak': 24, 'clic2020_test': 428, 'div2k_val': 100}, 'expected_counts': {'kodak': 24, 'clic2020_test': 428, 'div2k_val': 100}, 'count_status': {'kodak': 'ok', 'clic2020_test': 'ok', 'div2k_val': 'ok'}, 'total_images': 552, 'primary_metrics': ['PSNR', 'LPIPS', 'DISTS', 'FID'], 'bpp_metric': 'actual_payload_bpp for CoSER; official nominal/file bpp must be labeled for external codecs', 'default_crop_size': 512, 'notes': ['Legacy name retained for old experiment commands. Prefer cod_reproduction_512 for new runs.', 'Same dataset and 512x512 center-crop preprocessing as cod_reproduction_512.', 'Do not use this name for CoD-Lite full-resolution comparison tables.'], 'datasets': [{'key': 'kodak', 'display_name': 'Kodak 24', 'count': 24, 'expected_count': 24, 'status': 'ok', 'source_roots': ['/dpl/kodak'], 'first_path': '/dpl/kodak/kodim01.png', 'last_path': '/dpl/kodak/kodim24.png', 'notes': []}, {'key': 'clic2020_test', 'display_name': 'CLIC2020 test 428', 'count': 428, 'expected_count': 428, 'status': 'ok', 'source_roots': ['/dpl/clic/professional/test', '/dpl/clic/mobile/test'], 'first_path': '/dpl/clic/professional/test/01443654fdc57490d5c2d8b0fd9e065b.png', 'last_path': '/dpl/clic/mobile/test/ff52b7548f7024b96d2cea64e5e061b7.png', 'notes': ['professional/test and mobile/test are concatenated in that order.']}, {'key': 'div2k_val', 'display_name': 'DIV2K validation 100', 'count': 100, 'expected_count': 100, 'status': 'ok', 'source_roots': ['/dpl/div2k'], 'first_path': '/dpl/div2k/0801.png', 'last_path': '/dpl/div2k/0900.png', 'notes': ['Resolved by filename indices 0801-0900 from a mixed DIV2K root.']}]}
- deterministic_eval: True
- compute_perceptual: False
- save_reconstructions: True
- save_reconstruction_limit: 0
- save_reconstruction_triptychs: False
- save_decoder_feature_cache: True
- decoder_feature_cache_dir: results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/decoder_feature_cache
- residual_code: {'codec': 'static_residual_grid_semantic_position_leftctx_huffman', 'version': 0, 'bits': 4, 'value_range': 0.25, 'payload_codec': 'semantic_position_leftctx_huffman', 'detail_shape': [3, 16, 16], 'semantic_shape': [16, 16], 'group_count': 4, 'left_context_mode': 'start_left_sign4', 'left_context_count': 4, 'num_position_group_context_codes': 12288, 'min_code_length': 1, 'max_code_length': 14, 'mean_code_length_unweighted': 6.5782216389973955}
- semantic_payload_bpp_mean: 0.008997433427451313
- detail_payload_bpp_mean: 0.00500167625537817
- actual_payload_bpp_mean: 0.013999109682829483
- paper_bpp_mean: 0.013999109682829483
- total_payload_bpp_mean: 0.013999109682829483
- debug_semantic_only_full_stream_bpp_mean: 0.023462765458701313
- debug_full_stream_bpp_mean: 0.028891687807829483
- semantic_only_full_stream_bpp_mean: 0.023462765458701313
- stage3_full_stream_bpp_mean: 0.028891687807829483
- semantic_payload_bytes_mean: 294.8278985507246
- detail_payload_bytes_mean: 163.89492753623188
- stage3_stream_bytes_mean: 946.7228260869565
- semantic_only_psnr_mean: 21.56034038723379
- semantic_only_l1_mean: 0.05815943085379305
- semantic_only_ms_ssim_mean: 0.7302765013056173
- stage3_psnr_mean: 21.99658481107242
- stage3_l1_mean: 0.055934981874905636
- stage3_ms_ssim_mean: 0.7349218762736174
- semantic_topk_hit_rate_mean: 0.9351222826086957
- semantic_token_roundtrip_mean: 1.0
- detail_code_roundtrip_mean: 1.0
- residual_grid_abs_mean_mean: 0.01964723476650787
- residual_grid_std_mean: 0.024725589833721733
- residual_grid_clip_ratio_mean: 1.1794233443282977e-05
- detail_code_entropy_bits_mean: 1.6657354109313176
- stage3_psnr_delta_vs_semantic_only: 0.43624442383863027
- stage3_l1_delta_vs_semantic_only: -0.0022244489788874125
- stage3_ms_ssim_delta_vs_semantic_only: 0.004645374968000104
- all_semantic_tokens_roundtrip: True
- all_detail_codes_roundtrip: True
- roundtrip_failure_count: 0
- roundtrip_failures: []
- per_image_metrics: results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/per_image_metrics.jsonl
- reconstruction_manifest: results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl
- reconstruction_count: 552
- reconstruction_dirs: {'reference': 'results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/reference', 'semantic_only': 'results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/semantic_only', 'stage3': 'results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/stage3'}

## Artifacts

- summary: `results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/summary.json`
- per_image_metrics: `results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/per_image_metrics.jsonl`
- sample: `results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/stage3_uniform_residual_grid.png`
- reconstruction_manifest: `results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl`
