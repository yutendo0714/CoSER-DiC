# 20260627_stage3_b4_semposleft_g4_gencodec_reproduction_stage1_lpips002_ft500_tf384_topk1024_32kprior_8192resid_pp_unsharp200_perceptual

Date: 2026-06-27T20:50:55

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage3_uniform_residual_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_active5k_lpips002_fp32loss_ft500.pt --semantic-prior outputs/stage2_learned_entropy/20260627_stage2_learned_topk1024_escape_huffman_fit_32768tokens_stage1_lpips002_ft500_tf384_l6_12k/learned_topk_escape_huffman_prior.json --token-prior-checkpoint checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf384_l6_12kstep_32768tokens_stage1_lpips002_ft500_lr1e4_do02_es.pt --detail-prior outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_stage1_lpips002_ft500/static_residual_grid_semantic_position_leftctx_huffman_prior.json --eval-protocol gencodec_reproduction --detail-codec semantic_position_leftctx_huffman --detail-downsample-factor 32 --detail-bits 4 --detail-range 0.25 --decoder-postprocess unsharp3x3 --decoder-postprocess-strength 2.0 --stream-header-codec compact --stream-checksum-codec crc32 --compute-perceptual --wandb-mode offline --run-name 20260627_stage3_b4_semposleft_g4_gencodec_reproduction_stage1_lpips002_ft500_tf384_topk1024_32kprior_8192resid_pp_unsharp200_perceptual
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_active5k_lpips002_fp32loss_ft500.pt
- semantic_prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk1024_escape_huffman_fit_32768tokens_stage1_lpips002_ft500_tf384_l6_12k/learned_topk_escape_huffman_prior.json
- detail_prior: outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_stage1_lpips002_ft500/static_residual_grid_semantic_position_leftctx_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf384_l6_12kstep_32768tokens_stage1_lpips002_ft500_lr1e4_do02_es.pt
- num_images: 552
- crop_size: 256
- semantic_topk: 1024
- detail_downsample_factor: 32
- detail_shape: [3, 8, 8]
- detail_bits: 4
- detail_range: 0.25
- detail_gain: 1.0
- decoder_postprocess: unsharp3x3
- decoder_postprocess_strength: 2.0
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
- deterministic_eval: True
- compute_perceptual: True
- save_reconstructions: False
- save_reconstruction_limit: 0
- save_reconstruction_triptychs: False
- residual_code: {'codec': 'static_residual_grid_semantic_position_leftctx_huffman', 'version': 0, 'bits': 4, 'value_range': 0.25, 'payload_codec': 'semantic_position_leftctx_huffman', 'detail_shape': [3, 8, 8], 'semantic_shape': [8, 8], 'group_count': 4, 'left_context_mode': 'start_left_sign4', 'left_context_count': 4, 'num_position_group_context_codes': 3072, 'min_code_length': 1, 'max_code_length': 14, 'mean_code_length_unweighted': 6.48956298828125}
- semantic_payload_bpp_mean: 0.010235993758491848
- detail_payload_bpp_mean: 0.0051727294921875
- actual_payload_bpp_mean: 0.015408723250679348
- paper_bpp_mean: 0.015408723250679348
- total_payload_bpp_mean: 0.015408723250679348
- debug_semantic_only_full_stream_bpp_mean: 0.014996735945991848
- debug_full_stream_bpp_mean: 0.02468628814254982
- semantic_only_full_stream_bpp_mean: 0.014996735945991848
- stage3_full_stream_bpp_mean: 0.02468628814254982
- semantic_payload_bytes_mean: 83.85326086956522
- detail_payload_bytes_mean: 42.375
- stage3_stream_bytes_mean: 202.23007246376812
- semantic_only_psnr_mean: 21.549353012140248
- semantic_only_l1_mean: 0.06189432552344152
- semantic_only_ms_ssim_mean: 0.7197767667661327
- stage3_psnr_mean: 21.931414115256157
- stage3_l1_mean: 0.05938070170257403
- stage3_ms_ssim_mean: 0.7234825302671263
- semantic_topk_hit_rate_mean: 0.7692481884057971
- semantic_token_roundtrip_mean: 1.0
- detail_code_roundtrip_mean: 1.0
- residual_grid_abs_mean_mean: 0.020618243948590225
- residual_grid_std_mean: 0.025146802217510623
- residual_grid_clip_ratio_mean: 9.435386754626382e-06
- detail_code_entropy_bits_mean: 1.6463669557934222
- semantic_only_lpips_alex_mean: 0.5895735801086235
- semantic_only_dists_mean: 0.3820729705950488
- stage3_lpips_alex_mean: 0.5629108550121494
- stage3_dists_mean: 0.36449190596307535
- stage3_psnr_delta_vs_semantic_only: 0.3820611031159089
- stage3_l1_delta_vs_semantic_only: -0.0025136238208674916
- stage3_ms_ssim_delta_vs_semantic_only: 0.0037057635009936085
- stage3_lpips_alex_delta_vs_semantic_only: -0.02666272509647405
- stage3_dists_delta_vs_semantic_only: -0.017581064631973453
- all_semantic_tokens_roundtrip: True
- all_detail_codes_roundtrip: True
- roundtrip_failure_count: 0
- roundtrip_failures: []
- per_image_metrics: results/bitstreams/stage3_uniform_residual/20260627_stage3_b4_semposleft_g4_gencodec_reproduction_stage1_lpips002_ft500_tf384_topk1024_32kprior_8192resid_pp_unsharp200_perceptual/per_image_metrics.jsonl

## Artifacts

- summary: `results/bitstreams/stage3_uniform_residual/20260627_stage3_b4_semposleft_g4_gencodec_reproduction_stage1_lpips002_ft500_tf384_topk1024_32kprior_8192resid_pp_unsharp200_perceptual/summary.json`
- per_image_metrics: `results/bitstreams/stage3_uniform_residual/20260627_stage3_b4_semposleft_g4_gencodec_reproduction_stage1_lpips002_ft500_tf384_topk1024_32kprior_8192resid_pp_unsharp200_perceptual/per_image_metrics.jsonl`
- sample: `results/bitstreams/stage3_uniform_residual/20260627_stage3_b4_semposleft_g4_gencodec_reproduction_stage1_lpips002_ft500_tf384_topk1024_32kprior_8192resid_pp_unsharp200_perceptual/stage3_uniform_residual_grid.png`
