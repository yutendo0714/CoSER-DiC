# 20260627_stage3_b5_semposleft_g4_coser_common_lic_pp_unsharp200_recon_export

Date: 2026-06-27T17:36:15

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage3_uniform_residual_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --semantic-prior outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json --token-prior-checkpoint checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt --eval-protocol coser_common_lic --batch-size 4 --num-workers 4 --detail-downsample-factor 32 --detail-bits 5 --detail-range 0.25 --detail-gain 1.0 --detail-codec semantic_position_leftctx_huffman --detail-prior outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_8192calib_openimages_div2k/static_residual_grid_semantic_position_leftctx_huffman_prior.json --stream-header-codec compact --stream-checksum-codec crc32 --compute-perceptual --decoder-postprocess unsharp3x3 --decoder-postprocess-strength 2.00 --save-reconstructions --save-reconstruction-triptychs --run-name 20260627_stage3_b5_semposleft_g4_coser_common_lic_pp_unsharp200_recon_export
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- semantic_prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json
- detail_prior: outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_8192calib_openimages_div2k/static_residual_grid_semantic_position_leftctx_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt
- num_images: 165
- crop_size: 256
- semantic_topk: 512
- detail_downsample_factor: 32
- detail_shape: [3, 8, 8]
- detail_bits: 5
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
- save_reconstructions: True
- save_reconstruction_limit: 0
- save_reconstruction_triptychs: True
- residual_code: {'codec': 'static_residual_grid_semantic_position_leftctx_huffman', 'version': 0, 'bits': 5, 'value_range': 0.25, 'payload_codec': 'semantic_position_leftctx_huffman', 'detail_shape': [3, 8, 8], 'semantic_shape': [8, 8], 'group_count': 4, 'left_context_mode': 'start_left_sign4', 'left_context_count': 4, 'num_position_group_context_codes': 3072, 'min_code_length': 1, 'max_code_length': 18, 'mean_code_length_unweighted': 8.557993570963541}
- semantic_payload_bpp_mean: 0.010518022017045454
- detail_payload_bpp_mean: 0.007851710464015152
- actual_payload_bpp_mean: 0.018369732481060608
- paper_bpp_mean: 0.018369732481060608
- total_payload_bpp_mean: 0.018369732481060608
- debug_semantic_only_full_stream_bpp_mean: 0.015278764204545454
- debug_full_stream_bpp_mean: 0.027647816051136365
- semantic_only_full_stream_bpp_mean: 0.015278764204545454
- stage3_full_stream_bpp_mean: 0.027647816051136365
- semantic_payload_bytes_mean: 86.16363636363636
- detail_payload_bytes_mean: 64.32121212121213
- stage3_stream_bytes_mean: 226.4909090909091
- semantic_only_psnr_mean: 20.91873049880519
- semantic_only_l1_mean: 0.06730943628107057
- semantic_only_ms_ssim_mean: 0.6891279670325192
- stage3_psnr_mean: 21.344657516479494
- stage3_l1_mean: 0.06475576533738411
- stage3_ms_ssim_mean: 0.693968079848723
- semantic_topk_hit_rate_mean: 0.6509469696969697
- semantic_token_roundtrip_mean: 1.0
- detail_code_roundtrip_mean: 1.0
- residual_grid_abs_mean_mean: 0.02102516939242681
- residual_grid_std_mean: 0.025057107850796345
- residual_grid_clip_ratio_mean: 0.0
- detail_code_entropy_bits_mean: 2.549455948670705
- semantic_only_lpips_alex_mean: 0.7001213609043396
- semantic_only_dists_mean: 0.41472451795231213
- stage3_lpips_alex_mean: 0.6797586415922552
- stage3_dists_mean: 0.39923043503905786
- stage3_psnr_delta_vs_semantic_only: 0.4259270176743044
- stage3_l1_delta_vs_semantic_only: -0.0025536709436864563
- stage3_ms_ssim_delta_vs_semantic_only: 0.004840112816203779
- stage3_lpips_alex_delta_vs_semantic_only: -0.020362719312084443
- stage3_dists_delta_vs_semantic_only: -0.01549408291325427
- all_semantic_tokens_roundtrip: True
- all_detail_codes_roundtrip: True
- roundtrip_failure_count: 0
- roundtrip_failures: []
- per_image_metrics: results/bitstreams/stage3_uniform_residual/20260627_stage3_b5_semposleft_g4_coser_common_lic_pp_unsharp200_recon_export/per_image_metrics.jsonl
- reconstruction_manifest: results/bitstreams/stage3_uniform_residual/20260627_stage3_b5_semposleft_g4_coser_common_lic_pp_unsharp200_recon_export/reconstructions/manifest.jsonl
- reconstruction_count: 165
- reconstruction_dirs: {'reference': 'results/bitstreams/stage3_uniform_residual/20260627_stage3_b5_semposleft_g4_coser_common_lic_pp_unsharp200_recon_export/reconstructions/reference', 'semantic_only': 'results/bitstreams/stage3_uniform_residual/20260627_stage3_b5_semposleft_g4_coser_common_lic_pp_unsharp200_recon_export/reconstructions/semantic_only', 'stage3': 'results/bitstreams/stage3_uniform_residual/20260627_stage3_b5_semposleft_g4_coser_common_lic_pp_unsharp200_recon_export/reconstructions/stage3', 'triptych': 'results/bitstreams/stage3_uniform_residual/20260627_stage3_b5_semposleft_g4_coser_common_lic_pp_unsharp200_recon_export/reconstructions/triptych'}

## Artifacts

- summary: `results/bitstreams/stage3_uniform_residual/20260627_stage3_b5_semposleft_g4_coser_common_lic_pp_unsharp200_recon_export/summary.json`
- per_image_metrics: `results/bitstreams/stage3_uniform_residual/20260627_stage3_b5_semposleft_g4_coser_common_lic_pp_unsharp200_recon_export/per_image_metrics.jsonl`
- sample: `results/bitstreams/stage3_uniform_residual/20260627_stage3_b5_semposleft_g4_coser_common_lic_pp_unsharp200_recon_export/stage3_uniform_residual_grid.png`
- reconstruction_manifest: `results/bitstreams/stage3_uniform_residual/20260627_stage3_b5_semposleft_g4_coser_common_lic_pp_unsharp200_recon_export/reconstructions/manifest.jsonl`
