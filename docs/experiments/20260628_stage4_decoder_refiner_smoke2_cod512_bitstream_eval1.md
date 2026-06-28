# 20260628_stage4_decoder_refiner_smoke2_cod512_bitstream_eval1

Date: 2026-06-28T11:49:28

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage3_uniform_residual_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt --semantic-prior outputs/stage2_learned_entropy/20260628_stage2_learned_topk2048_escape_huffman_fit_32768tokens_crop512_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json --token-prior-checkpoint checkpoints/stage2_token_prior/20260628_stage2_token_prior_crop512_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_amp_es.pt --detail-codec semantic_position_leftctx_huffman --detail-prior outputs/stage3_residual_entropy/20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_crop512_8192calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json --eval-protocol cod_reproduction_512 --crop-size 512 --max-images 1 --batch-size 1 --num-workers 0 --detail-downsample-factor 32 --detail-bits 4 --detail-range 0.25 --detail-gain 0.5 --decoder-postprocess unsharp3x3 --decoder-postprocess-strength 0.15 --decoder-refiner-checkpoint checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_smoke2.pt --save-reconstructions --save-reconstruction-limit 1 --run-name 20260628_stage4_decoder_refiner_smoke2_cod512_bitstream_eval1 --wandb-mode offline
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt
- semantic_prior: outputs/stage2_learned_entropy/20260628_stage2_learned_topk2048_escape_huffman_fit_32768tokens_crop512_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json
- detail_prior: outputs/stage3_residual_entropy/20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_crop512_8192calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260628_stage2_token_prior_crop512_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_amp_es.pt
- num_images: 1
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
- decoder_refiner_checkpoint: checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_smoke2.pt
- decoder_refiner_enabled: True
- decoder_refiner_config: {'image_channels': 3, 'semantic_channels': 256, 'detail_channels': 3, 'base_channels': 64, 'semantic_context_channels': 32, 'num_res_blocks': 6, 'residual_scale': 0.125, 'use_semantic_latent': True, 'zero_init_output': True}
- decoder_refiner_payload_policy: fixed decoder-side weights; no additional actual_payload_bpp
- detail_codec: semantic_position_leftctx_huffman
- stream_header_codec: json
- stream_checksum_codec: sha256
- main_bpp_metric: actual_payload_bpp_mean
- paper_bpp_metric: paper_bpp_mean
- debug_bpp_metric: debug_full_stream_bpp_mean
- eval_protocol: cod_reproduction_512
- eval_datasets: []
- eval_image_roots: ['/dpl/kodak', '/dpl/clic/professional/test', '/dpl/clic/mobile/test', '/dpl/div2k']
- eval_protocol_summary: {'name': 'cod_reproduction_512', 'display_name': 'CoD paper reproduction protocol, 512x512', 'dataset_keys': ['kodak', 'clic2020_test', 'div2k_val'], 'dataset_counts': {'kodak': 24, 'clic2020_test': 428, 'div2k_val': 100}, 'expected_counts': {'kodak': 24, 'clic2020_test': 428, 'div2k_val': 100}, 'count_status': {'kodak': 'ok', 'clic2020_test': 'ok', 'div2k_val': 'ok'}, 'total_images': 552, 'primary_metrics': ['PSNR', 'LPIPS', 'DISTS', 'FID'], 'bpp_metric': 'actual_payload_bpp for CoSER; official nominal/file bpp must be labeled for external codecs', 'default_crop_size': 512, 'notes': ['Primary CoSER low-bitrate generative comparison protocol for CoD-style tables.', 'Resize if needed and center-crop every image to 512x512 before evaluation.', 'Use CLIC2020 test 428 as professional/test plus mobile/test.', 'Use DIV2K validation images 0801-0900, not the first 100 files from a mixed DIV2K root.', 'For CoD paper patch FID, run per dataset and label exact settings: Kodak512 uses 64px patches with split=2; CLIC2020 uses 128px patches with split=2.', 'DIV2K patch FID should be reported only when the chosen patch size/backend is explicitly labeled.', 'Patch-based or overlapped-patch FID must label patch size, shift/split count, and backend.'], 'datasets': [{'key': 'kodak', 'display_name': 'Kodak 24', 'count': 24, 'expected_count': 24, 'status': 'ok', 'source_roots': ['/dpl/kodak'], 'first_path': '/dpl/kodak/kodim01.png', 'last_path': '/dpl/kodak/kodim24.png', 'notes': []}, {'key': 'clic2020_test', 'display_name': 'CLIC2020 test 428', 'count': 428, 'expected_count': 428, 'status': 'ok', 'source_roots': ['/dpl/clic/professional/test', '/dpl/clic/mobile/test'], 'first_path': '/dpl/clic/professional/test/01443654fdc57490d5c2d8b0fd9e065b.png', 'last_path': '/dpl/clic/mobile/test/ff52b7548f7024b96d2cea64e5e061b7.png', 'notes': ['professional/test and mobile/test are concatenated in that order.']}, {'key': 'div2k_val', 'display_name': 'DIV2K validation 100', 'count': 100, 'expected_count': 100, 'status': 'ok', 'source_roots': ['/dpl/div2k'], 'first_path': '/dpl/div2k/0801.png', 'last_path': '/dpl/div2k/0900.png', 'notes': ['Resolved by filename indices 0801-0900 from a mixed DIV2K root.']}]}
- deterministic_eval: True
- compute_perceptual: False
- save_reconstructions: True
- save_reconstruction_limit: 1
- save_reconstruction_triptychs: False
- residual_code: {'codec': 'static_residual_grid_semantic_position_leftctx_huffman', 'version': 0, 'bits': 4, 'value_range': 0.25, 'payload_codec': 'semantic_position_leftctx_huffman', 'detail_shape': [3, 16, 16], 'semantic_shape': [16, 16], 'group_count': 4, 'left_context_mode': 'start_left_sign4', 'left_context_count': 4, 'num_position_group_context_codes': 12288, 'min_code_length': 1, 'max_code_length': 14, 'mean_code_length_unweighted': 6.5782216389973955}
- semantic_payload_bpp_mean: 0.010009765625
- detail_payload_bpp_mean: 0.00457763671875
- actual_payload_bpp_mean: 0.01458740234375
- paper_bpp_mean: 0.01458740234375
- total_payload_bpp_mean: 0.01458740234375
- debug_semantic_only_full_stream_bpp_mean: 0.02447509765625
- debug_full_stream_bpp_mean: 0.030181884765625
- semantic_only_full_stream_bpp_mean: 0.02447509765625
- stage3_full_stream_bpp_mean: 0.030181884765625
- semantic_payload_bytes_mean: 328.0
- detail_payload_bytes_mean: 150.0
- stage3_stream_bytes_mean: 989.0
- semantic_only_psnr_mean: 20.029088973999023
- semantic_only_l1_mean: 0.07279327511787415
- semantic_only_ms_ssim_mean: 0.5819405913352966
- stage3_psnr_mean: 20.141618728637695
- stage3_l1_mean: 0.07137744128704071
- stage3_ms_ssim_mean: 0.5851842164993286
- semantic_topk_hit_rate_mean: 0.890625
- semantic_token_roundtrip_mean: 1.0
- detail_code_roundtrip_mean: 1.0
- residual_grid_abs_mean_mean: 0.01560697890818119
- residual_grid_std_mean: 0.02023712731897831
- residual_grid_clip_ratio_mean: 0.0
- detail_code_entropy_bits_mean: 1.4605631828308105
- stage4_psnr_mean: 20.141164779663086
- stage4_l1_mean: 0.07138918340206146
- stage4_ms_ssim_mean: 0.5851799249649048
- stage4_refiner_residual_abs_mean_mean: 0.00022921606432646513
- stage3_psnr_delta_vs_semantic_only: 0.11252975463867188
- stage3_l1_delta_vs_semantic_only: -0.001415833830833435
- stage3_ms_ssim_delta_vs_semantic_only: 0.0032436251640319824
- stage4_psnr_delta_vs_stage3: -0.000453948974609375
- stage4_l1_delta_vs_stage3: 1.1742115020751953e-05
- stage4_ms_ssim_delta_vs_stage3: -4.291534423828125e-06
- all_semantic_tokens_roundtrip: True
- all_detail_codes_roundtrip: True
- roundtrip_failure_count: 0
- roundtrip_failures: []
- per_image_metrics: results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_smoke2_cod512_bitstream_eval1/per_image_metrics.jsonl
- reconstruction_manifest: results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_smoke2_cod512_bitstream_eval1/reconstructions/manifest.jsonl
- reconstruction_count: 1
- reconstruction_dirs: {'reference': 'results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_smoke2_cod512_bitstream_eval1/reconstructions/reference', 'semantic_only': 'results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_smoke2_cod512_bitstream_eval1/reconstructions/semantic_only', 'stage3': 'results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_smoke2_cod512_bitstream_eval1/reconstructions/stage3', 'stage4': 'results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_smoke2_cod512_bitstream_eval1/reconstructions/stage4'}

## Artifacts

- summary: `results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_smoke2_cod512_bitstream_eval1/summary.json`
- per_image_metrics: `results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_smoke2_cod512_bitstream_eval1/per_image_metrics.jsonl`
- sample: `results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_smoke2_cod512_bitstream_eval1/stage3_uniform_residual_grid.png`
- reconstruction_manifest: `results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_smoke2_cod512_bitstream_eval1/reconstructions/manifest.jsonl`
