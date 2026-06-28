# 20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_coser_common_pp_unsharp200_perceptual

Date: 2026-06-28T01:27:49

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage3_uniform_residual_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt --semantic-prior outputs/stage2_learned_entropy/20260628_stage2_learned_topk2048_escape_huffman_fit_32768tokens_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json --token-prior-checkpoint checkpoints/stage2_token_prior/20260628_stage2_token_prior_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_es.pt --detail-prior outputs/stage3_residual_entropy/20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json --eval-protocol coser_common_lic --batch-size 4 --num-workers 4 --detail-codec semantic_position_leftctx_huffman --detail-downsample-factor 32 --detail-bits 4 --detail-range 0.25 --decoder-postprocess unsharp3x3 --decoder-postprocess-strength 2.0 --stream-header-codec compact --stream-checksum-codec crc32 --compute-perceptual --wandb-mode offline --run-name 20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_coser_common_pp_unsharp200_perceptual
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt
- semantic_prior: outputs/stage2_learned_entropy/20260628_stage2_learned_topk2048_escape_huffman_fit_32768tokens_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json
- detail_prior: outputs/stage3_residual_entropy/20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260628_stage2_token_prior_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_es.pt
- num_images: 165
- crop_size: 256
- semantic_topk: 2048
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
- eval_protocol: coser_common_lic
- eval_datasets: []
- eval_image_roots: ['/dpl/kodak', '/dpl/clic/professional/valid', '/dpl/div2k']
- eval_protocol_summary: {'name': 'coser_common_lic', 'display_name': 'CoSER common LIC protocol', 'dataset_keys': ['kodak', 'clic_professional_valid', 'div2k_val'], 'dataset_counts': {'kodak': 24, 'clic_professional_valid': 41, 'div2k_val': 100}, 'expected_counts': {'kodak': 24, 'clic_professional_valid': 41, 'div2k_val': 100}, 'count_status': {'kodak': 'ok', 'clic_professional_valid': 'ok', 'div2k_val': 'ok'}, 'total_images': 165, 'primary_metrics': ['PSNR', 'MS-SSIM', 'LPIPS', 'DISTS', 'FID/KID when sample handling is defined'], 'bpp_metric': 'actual_payload_bpp', 'default_crop_size': None, 'notes': ['Internal CoSER LIC protocol for Kodak 24, CLIC Professional Validation 41, and DIV2K validation 100.', 'Do not merge this table with CLIC2020 test 428 comparisons.'], 'datasets': [{'key': 'kodak', 'display_name': 'Kodak 24', 'count': 24, 'expected_count': 24, 'status': 'ok', 'source_roots': ['/dpl/kodak'], 'first_path': '/dpl/kodak/kodim01.png', 'last_path': '/dpl/kodak/kodim24.png', 'notes': []}, {'key': 'clic_professional_valid', 'display_name': 'CLIC Professional Validation 41', 'count': 41, 'expected_count': 41, 'status': 'ok', 'source_roots': ['/dpl/clic/professional/valid'], 'first_path': '/dpl/clic/professional/valid/alberto-montalesi-176097.png', 'last_path': '/dpl/clic/professional/valid/zugr-108.png', 'notes': []}, {'key': 'div2k_val', 'display_name': 'DIV2K validation 100', 'count': 100, 'expected_count': 100, 'status': 'ok', 'source_roots': ['/dpl/div2k'], 'first_path': '/dpl/div2k/0801.png', 'last_path': '/dpl/div2k/0900.png', 'notes': ['Resolved by filename indices 0801-0900 from a mixed DIV2K root.']}]}
- deterministic_eval: True
- compute_perceptual: True
- save_reconstructions: False
- save_reconstruction_limit: 0
- save_reconstruction_triptychs: False
- residual_code: {'codec': 'static_residual_grid_semantic_position_leftctx_huffman', 'version': 0, 'bits': 4, 'value_range': 0.25, 'payload_codec': 'semantic_position_leftctx_huffman', 'detail_shape': [3, 8, 8], 'semantic_shape': [8, 8], 'group_count': 4, 'left_context_mode': 'start_left_sign4', 'left_context_count': 4, 'num_position_group_context_codes': 3072, 'min_code_length': 1, 'max_code_length': 14, 'mean_code_length_unweighted': 6.481160481770833}
- semantic_payload_bpp_mean: 0.010584605823863636
- detail_payload_bpp_mean: 0.005209812973484849
- actual_payload_bpp_mean: 0.015794418797348483
- paper_bpp_mean: 0.015794418797348483
- total_payload_bpp_mean: 0.015794418797348483
- debug_semantic_only_full_stream_bpp_mean: 0.015345348011363636
- debug_full_stream_bpp_mean: 0.025071762547348483
- semantic_only_full_stream_bpp_mean: 0.015345348011363636
- stage3_full_stream_bpp_mean: 0.025071762547348483
- semantic_payload_bytes_mean: 86.7090909090909
- detail_payload_bytes_mean: 42.67878787878788
- stage3_stream_bytes_mean: 205.38787878787878
- semantic_only_psnr_mean: 20.86489751989191
- semantic_only_l1_mean: 0.06745454658274398
- semantic_only_ms_ssim_mean: 0.690412453810374
- stage3_psnr_mean: 21.192536440762606
- stage3_l1_mean: 0.06554791053023303
- stage3_ms_ssim_mean: 0.693295748125423
- semantic_topk_hit_rate_mean: 0.8575757575757575
- semantic_token_roundtrip_mean: 1.0
- detail_code_roundtrip_mean: 1.0
- residual_grid_abs_mean_mean: 0.02107917455335458
- residual_grid_std_mean: 0.025425526084886355
- residual_grid_clip_ratio_mean: 0.0
- detail_code_entropy_bits_mean: 1.6772838713544789
- semantic_only_lpips_alex_mean: 0.6077778607381112
- semantic_only_dists_mean: 0.38456300894419354
- stage3_lpips_alex_mean: 0.5775717258001819
- stage3_dists_mean: 0.37190527518590294
- stage3_psnr_delta_vs_semantic_only: 0.3276389208706938
- stage3_l1_delta_vs_semantic_only: -0.0019066360525109571
- stage3_ms_ssim_delta_vs_semantic_only: 0.0028832943150490786
- stage3_lpips_alex_delta_vs_semantic_only: -0.03020613493792934
- stage3_dists_delta_vs_semantic_only: -0.012657733758290601
- all_semantic_tokens_roundtrip: True
- all_detail_codes_roundtrip: True
- roundtrip_failure_count: 0
- roundtrip_failures: []
- per_image_metrics: results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_coser_common_pp_unsharp200_perceptual/per_image_metrics.jsonl

## Artifacts

- summary: `results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_coser_common_pp_unsharp200_perceptual/summary.json`
- per_image_metrics: `results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_coser_common_pp_unsharp200_perceptual/per_image_metrics.jsonl`
- sample: `results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_coser_common_pp_unsharp200_perceptual/stage3_uniform_residual_grid.png`
