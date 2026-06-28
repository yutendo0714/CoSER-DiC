# 20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_cod512_gain075_pp_unsharp150_perceptual_recon

Date: 2026-06-28T10:05:04

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage3_uniform_residual_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt --semantic-prior outputs/stage2_learned_entropy/20260628_stage2_learned_topk2048_escape_huffman_fit_32768tokens_crop512_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json --token-prior-checkpoint checkpoints/stage2_token_prior/20260628_stage2_token_prior_crop512_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_amp_es.pt --detail-prior outputs/stage3_residual_entropy/20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_crop512_8192calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json --detail-codec semantic_position_leftctx_huffman --detail-downsample-factor 32 --detail-bits 4 --detail-range 0.25 --detail-gain 0.75 --decoder-postprocess unsharp3x3 --decoder-postprocess-strength 0.15 --eval-protocol cod_reproduction_512 --batch-size 16 --num-workers 8 --compute-perceptual --save-reconstructions --wandb-mode offline --run-name 20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_cod512_gain075_pp_unsharp150_perceptual_recon
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
- detail_gain: 0.75
- decoder_postprocess: unsharp3x3
- decoder_postprocess_strength: 0.15
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
- compute_perceptual: True
- save_reconstructions: True
- save_reconstruction_limit: 0
- save_reconstruction_triptychs: False
- residual_code: {'codec': 'static_residual_grid_semantic_position_leftctx_huffman', 'version': 0, 'bits': 4, 'value_range': 0.25, 'payload_codec': 'semantic_position_leftctx_huffman', 'detail_shape': [3, 16, 16], 'semantic_shape': [16, 16], 'group_count': 4, 'left_context_mode': 'start_left_sign4', 'left_context_count': 4, 'num_position_group_context_codes': 12288, 'min_code_length': 1, 'max_code_length': 14, 'mean_code_length_unweighted': 6.5782216389973955}
- semantic_payload_bpp_mean: 0.008997378141983696
- detail_payload_bpp_mean: 0.005002007968183877
- actual_payload_bpp_mean: 0.013999386110167572
- paper_bpp_mean: 0.013999386110167572
- total_payload_bpp_mean: 0.013999386110167572
- debug_semantic_only_full_stream_bpp_mean: 0.023462710173233696
- debug_full_stream_bpp_mean: 0.029593868532042572
- semantic_only_full_stream_bpp_mean: 0.023462710173233696
- stage3_full_stream_bpp_mean: 0.029593868532042572
- semantic_payload_bytes_mean: 294.82608695652175
- detail_payload_bytes_mean: 163.90579710144928
- stage3_stream_bytes_mean: 969.731884057971
- semantic_only_psnr_mean: 21.560301607933596
- semantic_only_l1_mean: 0.05815980814066648
- semantic_only_ms_ssim_mean: 0.7302760936832731
- stage3_psnr_mean: 21.965809749520343
- stage3_l1_mean: 0.055648825191251555
- stage3_ms_ssim_mean: 0.7349705171055984
- semantic_topk_hit_rate_mean: 0.9351293591485508
- semantic_token_roundtrip_mean: 1.0
- detail_code_roundtrip_mean: 1.0
- residual_grid_abs_mean_mean: 0.01964744698320367
- residual_grid_std_mean: 0.024725696213705385
- residual_grid_clip_ratio_mean: 1.1794233443282977e-05
- detail_code_entropy_bits_mean: 1.6658088419100512
- semantic_only_lpips_alex_mean: 0.5785916467818121
- semantic_only_dists_mean: 0.35850487422683963
- stage3_lpips_alex_mean: 0.5741776440603038
- stage3_dists_mean: 0.3547455971871597
- stage3_psnr_delta_vs_semantic_only: 0.405508141586747
- stage3_l1_delta_vs_semantic_only: -0.0025109829494149236
- stage3_ms_ssim_delta_vs_semantic_only: 0.004694423422325289
- stage3_lpips_alex_delta_vs_semantic_only: -0.00441400272150827
- stage3_dists_delta_vs_semantic_only: -0.0037592770396799313
- all_semantic_tokens_roundtrip: True
- all_detail_codes_roundtrip: True
- roundtrip_failure_count: 0
- roundtrip_failures: []
- per_image_metrics: results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_cod512_gain075_pp_unsharp150_perceptual_recon/per_image_metrics.jsonl
- reconstruction_manifest: results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_cod512_gain075_pp_unsharp150_perceptual_recon/reconstructions/manifest.jsonl
- reconstruction_count: 552
- reconstruction_dirs: {'reference': 'results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_cod512_gain075_pp_unsharp150_perceptual_recon/reconstructions/reference', 'semantic_only': 'results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_cod512_gain075_pp_unsharp150_perceptual_recon/reconstructions/semantic_only', 'stage3': 'results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_cod512_gain075_pp_unsharp150_perceptual_recon/reconstructions/stage3'}

## Artifacts

- summary: `results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_cod512_gain075_pp_unsharp150_perceptual_recon/summary.json`
- per_image_metrics: `results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_cod512_gain075_pp_unsharp150_perceptual_recon/per_image_metrics.jsonl`
- sample: `results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_cod512_gain075_pp_unsharp150_perceptual_recon/stage3_uniform_residual_grid.png`
- reconstruction_manifest: `results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_cod512_gain075_pp_unsharp150_perceptual_recon/reconstructions/manifest.jsonl`
