# 20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_anchor020_rs005_oi_only_clic64_eval

Date: 2026-06-28T12:24:09

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage3_uniform_residual_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt --semantic-prior outputs/stage2_learned_entropy/20260628_stage2_learned_topk2048_escape_huffman_fit_32768tokens_crop512_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json --token-prior-checkpoint checkpoints/stage2_token_prior/20260628_stage2_token_prior_crop512_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_amp_es.pt --detail-codec semantic_position_leftctx_huffman --detail-prior outputs/stage3_residual_entropy/20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_crop512_8192calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json --eval-protocol cod_reproduction_512 --eval-dataset clic2020_test --crop-size 512 --max-images 64 --batch-size 1 --detail-downsample-factor 32 --detail-bits 4 --detail-range 0.25 --detail-gain 0.5 --decoder-postprocess unsharp3x3 --decoder-postprocess-strength 0.15 --decoder-refiner-checkpoint checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_anchor020_rs005_oi_only.pt --compute-perceptual --save-reconstructions --save-reconstruction-limit 64 --run-name 20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_anchor020_rs005_oi_only_clic64_eval --wandb-mode offline
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt
- semantic_prior: outputs/stage2_learned_entropy/20260628_stage2_learned_topk2048_escape_huffman_fit_32768tokens_crop512_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json
- detail_prior: outputs/stage3_residual_entropy/20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_crop512_8192calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260628_stage2_token_prior_crop512_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_amp_es.pt
- num_images: 64
- crop_size: 512
- protocol_default_crop_size: 512
- crop_size_matches_protocol_default: True
- semantic_topk: 2048
- semantic_topk_schedule: prefix_replay_decoder_safe
- detail_downsample_factor: 32
- detail_shape: [3, 16, 16]
- detail_bits: 4
- detail_range: 0.25
- detail_gain: 0.5
- decoder_postprocess: unsharp3x3
- decoder_postprocess_strength: 0.15
- decoder_refiner_checkpoint: checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_anchor020_rs005_oi_only.pt
- decoder_refiner_enabled: True
- decoder_refiner_config: {'image_channels': 3, 'semantic_channels': 256, 'detail_channels': 3, 'base_channels': 64, 'semantic_context_channels': 32, 'num_res_blocks': 6, 'residual_scale': 0.05, 'use_semantic_latent': True, 'zero_init_output': True}
- decoder_refiner_payload_policy: fixed decoder-side weights; no additional actual_payload_bpp
- detail_codec: semantic_position_leftctx_huffman
- stream_header_codec: json
- stream_checksum_codec: sha256
- main_bpp_metric: actual_payload_bpp_mean
- paper_bpp_metric: paper_bpp_mean
- debug_bpp_metric: debug_full_stream_bpp_mean
- eval_protocol: cod_reproduction_512
- eval_datasets: ['clic2020_test']
- eval_image_roots: ['/dpl/clic/professional/test', '/dpl/clic/mobile/test']
- eval_protocol_summary: {'name': 'cod_reproduction_512', 'display_name': 'CoD paper reproduction protocol, 512x512', 'dataset_keys': ['clic2020_test'], 'dataset_counts': {'clic2020_test': 428}, 'expected_counts': {'clic2020_test': 428}, 'count_status': {'clic2020_test': 'ok'}, 'total_images': 428, 'primary_metrics': ['PSNR', 'LPIPS', 'DISTS', 'FID'], 'bpp_metric': 'actual_payload_bpp for CoSER; official nominal/file bpp must be labeled for external codecs', 'default_crop_size': 512, 'notes': ['Primary CoSER low-bitrate generative comparison protocol for CoD-style tables.', 'Resize if needed and center-crop every image to 512x512 before evaluation.', 'Use CLIC2020 test 428 as professional/test plus mobile/test.', 'Use DIV2K validation images 0801-0900, not the first 100 files from a mixed DIV2K root.', 'For CoD paper patch FID, run per dataset and label exact settings: Kodak512 uses 64px patches with split=2; CLIC2020 uses 128px patches with split=2.', 'DIV2K patch FID should be reported only when the chosen patch size/backend is explicitly labeled.', 'Patch-based or overlapped-patch FID must label patch size, shift/split count, and backend.'], 'datasets': [{'key': 'clic2020_test', 'display_name': 'CLIC2020 test 428', 'count': 428, 'expected_count': 428, 'status': 'ok', 'source_roots': ['/dpl/clic/professional/test', '/dpl/clic/mobile/test'], 'first_path': '/dpl/clic/professional/test/01443654fdc57490d5c2d8b0fd9e065b.png', 'last_path': '/dpl/clic/mobile/test/ff52b7548f7024b96d2cea64e5e061b7.png', 'notes': ['professional/test and mobile/test are concatenated in that order.']}]}
- deterministic_eval: True
- compute_perceptual: True
- save_reconstructions: True
- save_reconstruction_limit: 64
- save_reconstruction_triptychs: False
- residual_code: {'codec': 'static_residual_grid_semantic_position_leftctx_huffman', 'version': 0, 'bits': 4, 'value_range': 0.25, 'payload_codec': 'semantic_position_leftctx_huffman', 'detail_shape': [3, 16, 16], 'semantic_shape': [16, 16], 'group_count': 4, 'left_context_mode': 'start_left_sign4', 'left_context_count': 4, 'num_position_group_context_codes': 12288, 'min_code_length': 1, 'max_code_length': 14, 'mean_code_length_unweighted': 6.5782216389973955}
- semantic_payload_bpp_mean: 0.008575916290283203
- detail_payload_bpp_mean: 0.005017280578613281
- actual_payload_bpp_mean: 0.013593196868896484
- paper_bpp_mean: 0.013593196868896484
- total_payload_bpp_mean: 0.013593196868896484
- debug_semantic_only_full_stream_bpp_mean: 0.023041248321533203
- debug_full_stream_bpp_mean: 0.029187679290771484
- semantic_only_full_stream_bpp_mean: 0.023041248321533203
- stage3_full_stream_bpp_mean: 0.029187679290771484
- semantic_payload_bytes_mean: 281.015625
- detail_payload_bytes_mean: 164.40625
- stage3_stream_bytes_mean: 956.421875
- semantic_only_psnr_mean: 22.002647414803505
- semantic_only_l1_mean: 0.053595605757436715
- semantic_only_ms_ssim_mean: 0.7690864000469446
- stage3_psnr_mean: 22.3812445551157
- stage3_l1_mean: 0.050973843535757624
- stage3_ms_ssim_mean: 0.772970337420702
- semantic_topk_hit_rate_mean: 0.943115234375
- semantic_token_roundtrip_mean: 1.0
- detail_code_roundtrip_mean: 1.0
- residual_grid_abs_mean_mean: 0.01944168226327747
- residual_grid_std_mean: 0.025350934913149104
- residual_grid_clip_ratio_mean: 2.0345052689663135e-05
- detail_code_entropy_bits_mean: 1.6798672592267394
- semantic_only_lpips_alex_mean: 0.5048623570473865
- semantic_only_dists_mean: 0.33925055246800184
- stage3_lpips_alex_mean: 0.49825221527135
- stage3_dists_mean: 0.33619323186576366
- stage4_psnr_mean: 22.302455604076385
- stage4_l1_mean: 0.05137174670380773
- stage4_ms_ssim_mean: 0.7700303178280592
- stage4_refiner_residual_abs_mean_mean: 0.003626723120760289
- stage4_lpips_alex_mean: 0.4700946604134515
- stage4_dists_mean: 0.322704934515059
- stage3_psnr_delta_vs_semantic_only: 0.3785971403121948
- stage3_l1_delta_vs_semantic_only: -0.0026217622216790915
- stage3_ms_ssim_delta_vs_semantic_only: 0.0038839373737573624
- stage3_lpips_alex_delta_vs_semantic_only: -0.006610141776036471
- stage3_dists_delta_vs_semantic_only: -0.0030573206022381783
- stage4_psnr_delta_vs_stage3: -0.07878895103931427
- stage4_l1_delta_vs_stage3: 0.0003979031680501066
- stage4_ms_ssim_delta_vs_stage3: -0.002940019592642784
- stage4_lpips_alex_delta_vs_stage3: -0.028157554857898504
- stage4_dists_delta_vs_stage3: -0.01348829735070467
- all_semantic_tokens_roundtrip: True
- all_detail_codes_roundtrip: True
- roundtrip_failure_count: 0
- roundtrip_failures: []
- per_image_metrics: results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_anchor020_rs005_oi_only_clic64_eval/per_image_metrics.jsonl
- reconstruction_manifest: results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_anchor020_rs005_oi_only_clic64_eval/reconstructions/manifest.jsonl
- reconstruction_count: 64
- reconstruction_dirs: {'reference': 'results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_anchor020_rs005_oi_only_clic64_eval/reconstructions/reference', 'semantic_only': 'results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_anchor020_rs005_oi_only_clic64_eval/reconstructions/semantic_only', 'stage3': 'results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_anchor020_rs005_oi_only_clic64_eval/reconstructions/stage3', 'stage4': 'results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_anchor020_rs005_oi_only_clic64_eval/reconstructions/stage4'}

## Artifacts

- summary: `results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_anchor020_rs005_oi_only_clic64_eval/summary.json`
- per_image_metrics: `results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_anchor020_rs005_oi_only_clic64_eval/per_image_metrics.jsonl`
- sample: `results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_anchor020_rs005_oi_only_clic64_eval/stage3_uniform_residual_grid.png`
- reconstruction_manifest: `results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_anchor020_rs005_oi_only_clic64_eval/reconstructions/manifest.jsonl`
