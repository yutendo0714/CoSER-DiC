# 20260628_stage3_b16_topk512_tf384_semposleft_g4_b4_gencodec512_smoke1_no_pp_prefixsafe

Date: 2026-06-28T03:02:38

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage3_uniform_residual_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt --semantic-prior outputs/stage2_learned_entropy/20260628_stage2_learned_topk512_escape_huffman_fit_32768tokens_crop512_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json --detail-prior outputs/stage3_residual_entropy/20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_crop512_8192calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json --eval-protocol gencodec_reproduction --crop-size 512 --max-images 1 --batch-size 1 --num-workers 2 --detail-downsample-factor 32 --detail-bits 4 --detail-range 0.25 --detail-codec semantic_position_leftctx_huffman --wandb-mode offline --run-name 20260628_stage3_b16_topk512_tf384_semposleft_g4_b4_gencodec512_smoke1_no_pp_prefixsafe
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt
- semantic_prior: outputs/stage2_learned_entropy/20260628_stage2_learned_topk512_escape_huffman_fit_32768tokens_crop512_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json
- detail_prior: outputs/stage3_residual_entropy/20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_crop512_8192calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260628_stage2_token_prior_crop512_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_amp_es.pt
- num_images: 1
- crop_size: 512
- protocol_default_crop_size: 512
- crop_size_matches_protocol_default: True
- semantic_topk: 512
- detail_downsample_factor: 32
- detail_shape: [3, 16, 16]
- detail_bits: 4
- detail_range: 0.25
- detail_gain: 1.0
- decoder_postprocess: none
- decoder_postprocess_strength: 0.0
- detail_codec: semantic_position_leftctx_huffman
- stream_header_codec: json
- stream_checksum_codec: sha256
- main_bpp_metric: actual_payload_bpp_mean
- paper_bpp_metric: paper_bpp_mean
- debug_bpp_metric: debug_full_stream_bpp_mean
- eval_protocol: gencodec_reproduction
- eval_datasets: []
- eval_image_roots: ['/dpl/kodak', '/dpl/clic/professional/test', '/dpl/clic/mobile/test', '/dpl/div2k']
- eval_protocol_summary: {'name': 'gencodec_reproduction', 'display_name': 'GenCodec / CoD / CoD-Lite reproduction protocol', 'dataset_keys': ['kodak', 'clic2020_test', 'div2k_val'], 'dataset_counts': {'kodak': 24, 'clic2020_test': 428, 'div2k_val': 100}, 'expected_counts': {'kodak': 24, 'clic2020_test': 428, 'div2k_val': 100}, 'count_status': {'kodak': 'ok', 'clic2020_test': 'ok', 'div2k_val': 'ok'}, 'total_images': 552, 'primary_metrics': ['PSNR', 'LPIPS', 'DISTS', 'FID'], 'bpp_metric': 'actual_payload_bpp for CoSER; official nominal/file bpp must be labeled for external codecs', 'default_crop_size': 512, 'notes': ['Use CLIC2020 test 428 as professional/test plus mobile/test.', 'Use DIV2K validation images 0801-0900, not the first 100 files from a mixed DIV2K root.', 'Patch-based or overlapped-patch FID must be labeled when matching CoD/CoD-Lite.'], 'datasets': [{'key': 'kodak', 'display_name': 'Kodak 24', 'count': 24, 'expected_count': 24, 'status': 'ok', 'source_roots': ['/dpl/kodak'], 'first_path': '/dpl/kodak/kodim01.png', 'last_path': '/dpl/kodak/kodim24.png', 'notes': []}, {'key': 'clic2020_test', 'display_name': 'CLIC2020 test 428', 'count': 428, 'expected_count': 428, 'status': 'ok', 'source_roots': ['/dpl/clic/professional/test', '/dpl/clic/mobile/test'], 'first_path': '/dpl/clic/professional/test/01443654fdc57490d5c2d8b0fd9e065b.png', 'last_path': '/dpl/clic/mobile/test/ff52b7548f7024b96d2cea64e5e061b7.png', 'notes': ['professional/test and mobile/test are concatenated in that order.']}, {'key': 'div2k_val', 'display_name': 'DIV2K validation 100', 'count': 100, 'expected_count': 100, 'status': 'ok', 'source_roots': ['/dpl/div2k'], 'first_path': '/dpl/div2k/0801.png', 'last_path': '/dpl/div2k/0900.png', 'notes': ['Resolved by filename indices 0801-0900 from a mixed DIV2K root.']}]}
- deterministic_eval: True
- compute_perceptual: False
- save_reconstructions: False
- save_reconstruction_limit: 0
- save_reconstruction_triptychs: False
- residual_code: {'codec': 'static_residual_grid_semantic_position_leftctx_huffman', 'version': 0, 'bits': 4, 'value_range': 0.25, 'payload_codec': 'semantic_position_leftctx_huffman', 'detail_shape': [3, 16, 16], 'semantic_shape': [16, 16], 'group_count': 4, 'left_context_mode': 'start_left_sign4', 'left_context_count': 4, 'num_position_group_context_codes': 12288, 'min_code_length': 1, 'max_code_length': 14, 'mean_code_length_unweighted': 6.5782216389973955}
- semantic_payload_bpp_mean: 0.01019287109375
- detail_payload_bpp_mean: 0.00457763671875
- actual_payload_bpp_mean: 0.0147705078125
- paper_bpp_mean: 0.0147705078125
- total_payload_bpp_mean: 0.0147705078125
- debug_semantic_only_full_stream_bpp_mean: 0.024658203125
- debug_full_stream_bpp_mean: 0.0296630859375
- semantic_only_full_stream_bpp_mean: 0.024658203125
- stage3_full_stream_bpp_mean: 0.0296630859375
- semantic_payload_bytes_mean: 334.0
- detail_payload_bytes_mean: 150.0
- stage3_stream_bytes_mean: 972.0
- semantic_only_psnr_mean: 20.029088973999023
- semantic_only_l1_mean: 0.07279327511787415
- semantic_only_ms_ssim_mean: 0.5819405913352966
- stage3_psnr_mean: 20.181066513061523
- stage3_l1_mean: 0.07084843516349792
- stage3_ms_ssim_mean: 0.5862910747528076
- semantic_topk_hit_rate_mean: 0.66796875
- semantic_token_roundtrip_mean: 1.0
- detail_code_roundtrip_mean: 1.0
- residual_grid_abs_mean_mean: 0.01560697890818119
- residual_grid_std_mean: 0.02023712731897831
- residual_grid_clip_ratio_mean: 0.0
- detail_code_entropy_bits_mean: 1.4605631828308105
- stage3_psnr_delta_vs_semantic_only: 0.1519775390625
- stage3_l1_delta_vs_semantic_only: -0.0019448399543762207
- stage3_ms_ssim_delta_vs_semantic_only: 0.004350483417510986
- all_semantic_tokens_roundtrip: True
- all_detail_codes_roundtrip: True
- roundtrip_failure_count: 0
- roundtrip_failures: []
- per_image_metrics: results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk512_tf384_semposleft_g4_b4_gencodec512_smoke1_no_pp_prefixsafe/per_image_metrics.jsonl

## Artifacts

- summary: `results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk512_tf384_semposleft_g4_b4_gencodec512_smoke1_no_pp_prefixsafe/summary.json`
- per_image_metrics: `results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk512_tf384_semposleft_g4_b4_gencodec512_smoke1_no_pp_prefixsafe/per_image_metrics.jsonl`
- sample: `results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk512_tf384_semposleft_g4_b4_gencodec512_smoke1_no_pp_prefixsafe/stage3_uniform_residual_grid.png`
