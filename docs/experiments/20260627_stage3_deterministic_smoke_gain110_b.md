# 20260627_stage3_deterministic_smoke_gain110_b

Date: 2026-06-27T17:13:25

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage3_uniform_residual_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --semantic-prior outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json --token-prior-checkpoint checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt --eval-protocol coser_common_lic --eval-dataset kodak --max-images 8 --batch-size 4 --num-workers 2 --detail-downsample-factor 32 --detail-bits 5 --detail-range 0.25 --detail-gain 1.1 --detail-codec semantic_position_leftctx_huffman --detail-prior outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_8192calib_openimages_div2k/static_residual_grid_semantic_position_leftctx_huffman_prior.json --stream-header-codec compact --stream-checksum-codec crc32 --run-name 20260627_stage3_deterministic_smoke_gain110_b
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- semantic_prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json
- detail_prior: outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_8192calib_openimages_div2k/static_residual_grid_semantic_position_leftctx_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt
- num_images: 8
- crop_size: 256
- semantic_topk: 512
- detail_downsample_factor: 32
- detail_shape: [3, 8, 8]
- detail_bits: 5
- detail_range: 0.25
- detail_gain: 1.1
- detail_codec: semantic_position_leftctx_huffman
- stream_header_codec: compact
- stream_checksum_codec: crc32
- main_bpp_metric: actual_payload_bpp_mean
- paper_bpp_metric: paper_bpp_mean
- debug_bpp_metric: debug_full_stream_bpp_mean
- eval_protocol: coser_common_lic
- eval_datasets: ['kodak']
- eval_image_roots: ['/dpl/kodak']
- eval_protocol_summary: {'name': 'coser_common_lic', 'display_name': 'CoSER common LIC protocol', 'dataset_keys': ['kodak'], 'dataset_counts': {'kodak': 24}, 'expected_counts': {'kodak': 24}, 'count_status': {'kodak': 'ok'}, 'total_images': 24, 'primary_metrics': ['PSNR', 'MS-SSIM', 'LPIPS', 'DISTS', 'FID/KID when sample handling is defined'], 'bpp_metric': 'actual_payload_bpp', 'default_crop_size': None, 'notes': ['Internal CoSER LIC protocol for Kodak 24, CLIC Professional Validation 41, and DIV2K validation 100.', 'Do not merge this table with CLIC2020 test 428 comparisons.'], 'datasets': [{'key': 'kodak', 'display_name': 'Kodak 24', 'count': 24, 'expected_count': 24, 'status': 'ok', 'source_roots': ['/dpl/kodak'], 'first_path': '/dpl/kodak/kodim01.png', 'last_path': '/dpl/kodak/kodim24.png', 'notes': []}]}
- deterministic_eval: True
- compute_perceptual: False
- save_reconstructions: False
- save_reconstruction_limit: 0
- save_reconstruction_triptychs: False
- residual_code: {'codec': 'static_residual_grid_semantic_position_leftctx_huffman', 'version': 0, 'bits': 5, 'value_range': 0.25, 'payload_codec': 'semantic_position_leftctx_huffman', 'detail_shape': [3, 8, 8], 'semantic_shape': [8, 8], 'group_count': 4, 'left_context_mode': 'start_left_sign4', 'left_context_count': 4, 'num_position_group_context_codes': 3072, 'min_code_length': 1, 'max_code_length': 18, 'mean_code_length_unweighted': 8.557993570963541}
- semantic_payload_bpp_mean: 0.010986328125
- detail_payload_bpp_mean: 0.007965087890625
- actual_payload_bpp_mean: 0.018951416015625
- paper_bpp_mean: 0.018951416015625
- total_payload_bpp_mean: 0.018951416015625
- debug_semantic_only_full_stream_bpp_mean: 0.0157470703125
- debug_full_stream_bpp_mean: 0.026885986328125
- semantic_only_full_stream_bpp_mean: 0.0157470703125
- stage3_full_stream_bpp_mean: 0.026885986328125
- semantic_payload_bytes_mean: 90.0
- detail_payload_bytes_mean: 65.25
- stage3_stream_bytes_mean: 220.25
- semantic_only_psnr_mean: 20.453500747680664
- semantic_only_l1_mean: 0.07213027123361826
- semantic_only_ms_ssim_mean: 0.6378082893788815
- stage3_psnr_mean: 20.908830165863037
- stage3_l1_mean: 0.06907673692330718
- stage3_ms_ssim_mean: 0.643843624740839
- semantic_topk_hit_rate_mean: 0.58984375
- semantic_token_roundtrip_mean: 1.0
- detail_code_roundtrip_mean: 1.0
- residual_grid_abs_mean_mean: 0.021659598336555064
- residual_grid_std_mean: 0.025766815524548292
- residual_grid_clip_ratio_mean: 0.0
- detail_code_entropy_bits_mean: 2.619546875357628
- stage3_psnr_delta_vs_semantic_only: 0.45532941818237305
- stage3_l1_delta_vs_semantic_only: -0.003053534310311079
- stage3_ms_ssim_delta_vs_semantic_only: 0.00603533536195755
- all_semantic_tokens_roundtrip: True
- all_detail_codes_roundtrip: True
- roundtrip_failure_count: 0
- roundtrip_failures: []
- per_image_metrics: results/bitstreams/stage3_uniform_residual/20260627_stage3_deterministic_smoke_gain110_b/per_image_metrics.jsonl

## Artifacts

- summary: `results/bitstreams/stage3_uniform_residual/20260627_stage3_deterministic_smoke_gain110_b/summary.json`
- per_image_metrics: `results/bitstreams/stage3_uniform_residual/20260627_stage3_deterministic_smoke_gain110_b/per_image_metrics.jsonl`
- sample: `results/bitstreams/stage3_uniform_residual/20260627_stage3_deterministic_smoke_gain110_b/stage3_uniform_residual_grid.png`
