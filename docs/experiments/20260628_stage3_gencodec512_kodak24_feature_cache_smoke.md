# 20260628_stage3_gencodec512_kodak24_feature_cache_smoke

Date: 2026-06-28T17:55:18

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage3_uniform_residual_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt --semantic-prior outputs/stage2_learned_entropy/20260628_stage2_learned_topk2048_escape_huffman_fit_32768tokens_crop512_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json --token-prior-checkpoint checkpoints/stage2_token_prior/20260628_stage2_token_prior_crop512_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_amp_es.pt --detail-prior outputs/stage3_residual_entropy/20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_crop512_8192calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json --eval-protocol gencodec_reproduction --eval-dataset kodak --crop-size 512 --batch-size 4 --num-workers 4 --detail-downsample-factor 32 --detail-bits 4 --detail-range 0.25 --detail-gain 1.0 --detail-codec semantic_position_leftctx_huffman --stream-header-codec json --stream-checksum-codec sha256 --save-reconstructions --save-decoder-feature-cache --output-dir results/bitstreams/stage3_uniform_residual --run-name 20260628_stage3_gencodec512_kodak24_feature_cache_smoke --wandb-mode offline
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt
- semantic_prior: outputs/stage2_learned_entropy/20260628_stage2_learned_topk2048_escape_huffman_fit_32768tokens_crop512_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json
- detail_prior: outputs/stage3_residual_entropy/20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_crop512_8192calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260628_stage2_token_prior_crop512_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_amp_es.pt
- num_images: 24
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
- eval_datasets: ['kodak']
- eval_image_roots: ['/dpl/kodak']
- eval_protocol_summary: {'name': 'gencodec_reproduction', 'display_name': 'Legacy alias for CoD 512 reproduction protocol', 'dataset_keys': ['kodak'], 'dataset_counts': {'kodak': 24}, 'expected_counts': {'kodak': 24}, 'count_status': {'kodak': 'ok'}, 'total_images': 24, 'primary_metrics': ['PSNR', 'LPIPS', 'DISTS', 'FID'], 'bpp_metric': 'actual_payload_bpp for CoSER; official nominal/file bpp must be labeled for external codecs', 'default_crop_size': 512, 'notes': ['Legacy name retained for old experiment commands. Prefer cod_reproduction_512 for new runs.', 'Same dataset and 512x512 center-crop preprocessing as cod_reproduction_512.', 'Do not use this name for CoD-Lite full-resolution comparison tables.'], 'datasets': [{'key': 'kodak', 'display_name': 'Kodak 24', 'count': 24, 'expected_count': 24, 'status': 'ok', 'source_roots': ['/dpl/kodak'], 'first_path': '/dpl/kodak/kodim01.png', 'last_path': '/dpl/kodak/kodim24.png', 'notes': []}]}
- deterministic_eval: True
- compute_perceptual: False
- save_reconstructions: True
- save_reconstruction_limit: 0
- save_reconstruction_triptychs: False
- save_decoder_feature_cache: True
- decoder_feature_cache_dir: results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_kodak24_feature_cache_smoke/decoder_feature_cache
- residual_code: {'codec': 'static_residual_grid_semantic_position_leftctx_huffman', 'version': 0, 'bits': 4, 'value_range': 0.25, 'payload_codec': 'semantic_position_leftctx_huffman', 'detail_shape': [3, 16, 16], 'semantic_shape': [16, 16], 'group_count': 4, 'left_context_mode': 'start_left_sign4', 'left_context_count': 4, 'num_position_group_context_codes': 12288, 'min_code_length': 1, 'max_code_length': 14, 'mean_code_length_unweighted': 6.5782216389973955}
- semantic_payload_bpp_mean: 0.009375254313151041
- detail_payload_bpp_mean: 0.004748026529947917
- actual_payload_bpp_mean: 0.014123280843098959
- paper_bpp_mean: 0.014123280843098959
- total_payload_bpp_mean: 0.014123280843098959
- debug_semantic_only_full_stream_bpp_mean: 0.023840586344401043
- debug_full_stream_bpp_mean: 0.029015858968098957
- semantic_only_full_stream_bpp_mean: 0.023840586344401043
- stage3_full_stream_bpp_mean: 0.029015858968098957
- semantic_payload_bytes_mean: 307.2083333333333
- detail_payload_bytes_mean: 155.58333333333334
- stage3_stream_bytes_mean: 950.7916666666666
- semantic_only_psnr_mean: 21.309279680252075
- semantic_only_l1_mean: 0.05731871568908294
- semantic_only_ms_ssim_mean: 0.7173226674397787
- stage3_psnr_mean: 21.66828767458598
- stage3_l1_mean: 0.05430419510230422
- stage3_ms_ssim_mean: 0.7226272920767466
- semantic_topk_hit_rate_mean: 0.9138997395833334
- semantic_token_roundtrip_mean: 1.0
- detail_code_roundtrip_mean: 1.0
- residual_grid_abs_mean_mean: 0.01819253743936618
- residual_grid_std_mean: 0.023480882906975847
- residual_grid_clip_ratio_mean: 0.0
- detail_code_entropy_bits_mean: 1.6157998740673065
- stage3_psnr_delta_vs_semantic_only: 0.3590079943339042
- stage3_l1_delta_vs_semantic_only: -0.0030145205867787225
- stage3_ms_ssim_delta_vs_semantic_only: 0.005304624636967903
- all_semantic_tokens_roundtrip: True
- all_detail_codes_roundtrip: True
- roundtrip_failure_count: 0
- roundtrip_failures: []
- per_image_metrics: results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_kodak24_feature_cache_smoke/per_image_metrics.jsonl
- reconstruction_manifest: results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_kodak24_feature_cache_smoke/reconstructions/manifest.jsonl
- reconstruction_count: 24
- reconstruction_dirs: {'reference': 'results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_kodak24_feature_cache_smoke/reconstructions/reference', 'semantic_only': 'results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_kodak24_feature_cache_smoke/reconstructions/semantic_only', 'stage3': 'results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_kodak24_feature_cache_smoke/reconstructions/stage3'}

## Artifacts

- summary: `results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_kodak24_feature_cache_smoke/summary.json`
- per_image_metrics: `results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_kodak24_feature_cache_smoke/per_image_metrics.jsonl`
- sample: `results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_kodak24_feature_cache_smoke/stage3_uniform_residual_grid.png`
- reconstruction_manifest: `results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_kodak24_feature_cache_smoke/reconstructions/manifest.jsonl`
