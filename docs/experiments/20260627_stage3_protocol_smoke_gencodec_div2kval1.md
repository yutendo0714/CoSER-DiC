# 20260627_stage3_protocol_smoke_gencodec_div2kval1

Date: 2026-06-27T16:16:46

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage3_uniform_residual_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --semantic-prior outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json --token-prior-checkpoint checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt --eval-protocol gencodec_reproduction --eval-dataset div2k_val --max-images 1 --crop-size 256 --batch-size 1 --num-workers 0 --detail-downsample-factor 32 --detail-bits 5 --detail-range 0.25 --detail-codec semantic_position_leftctx_huffman --detail-prior outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_8192calib_openimages_div2k/static_residual_grid_semantic_position_leftctx_huffman_prior.json --stream-header-codec compact --stream-checksum-codec crc32 --wandb-mode offline --run-name 20260627_stage3_protocol_smoke_gencodec_div2kval1
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- semantic_prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json
- detail_prior: outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b5_r025_8192calib_openimages_div2k/static_residual_grid_semantic_position_leftctx_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt
- num_images: 1
- crop_size: 256
- semantic_topk: 512
- detail_downsample_factor: 32
- detail_shape: [3, 8, 8]
- detail_bits: 5
- detail_range: 0.25
- detail_codec: semantic_position_leftctx_huffman
- stream_header_codec: compact
- stream_checksum_codec: crc32
- main_bpp_metric: actual_payload_bpp_mean
- paper_bpp_metric: paper_bpp_mean
- debug_bpp_metric: debug_full_stream_bpp_mean
- eval_protocol: gencodec_reproduction
- eval_datasets: ['div2k_val']
- eval_image_roots: ['/dpl/div2k']
- eval_protocol_summary: {'name': 'gencodec_reproduction', 'display_name': 'GenCodec / CoD / CoD-Lite reproduction protocol', 'dataset_keys': ['div2k_val'], 'dataset_counts': {'div2k_val': 100}, 'expected_counts': {'div2k_val': 100}, 'count_status': {'div2k_val': 'ok'}, 'total_images': 100, 'primary_metrics': ['PSNR', 'LPIPS', 'DISTS', 'FID'], 'bpp_metric': 'actual_payload_bpp for CoSER; official nominal/file bpp must be labeled for external codecs', 'default_crop_size': 512, 'notes': ['Use CLIC2020 test 428 as professional/test plus mobile/test.', 'Use DIV2K validation images 0801-0900, not the first 100 files from a mixed DIV2K root.', 'Patch-based or overlapped-patch FID must be labeled when matching CoD/CoD-Lite.'], 'datasets': [{'key': 'div2k_val', 'display_name': 'DIV2K validation 100', 'count': 100, 'expected_count': 100, 'status': 'ok', 'source_roots': ['/dpl/div2k'], 'first_path': '/dpl/div2k/0801.png', 'last_path': '/dpl/div2k/0900.png', 'notes': ['Resolved by filename indices 0801-0900 from a mixed DIV2K root.']}]}
- residual_code: {'codec': 'static_residual_grid_semantic_position_leftctx_huffman', 'version': 0, 'bits': 5, 'value_range': 0.25, 'payload_codec': 'semantic_position_leftctx_huffman', 'detail_shape': [3, 8, 8], 'semantic_shape': [8, 8], 'group_count': 4, 'left_context_mode': 'start_left_sign4', 'left_context_count': 4, 'num_position_group_context_codes': 3072, 'min_code_length': 1, 'max_code_length': 18, 'mean_code_length_unweighted': 8.557993570963541}
- semantic_payload_bpp_mean: 0.0098876953125
- detail_payload_bpp_mean: 0.0064697265625
- actual_payload_bpp_mean: 0.016357421875
- paper_bpp_mean: 0.016357421875
- total_payload_bpp_mean: 0.016357421875
- debug_semantic_only_full_stream_bpp_mean: 0.0146484375
- debug_full_stream_bpp_mean: 0.0213623046875
- semantic_only_full_stream_bpp_mean: 0.0146484375
- stage3_full_stream_bpp_mean: 0.0213623046875
- semantic_payload_bytes_mean: 81.0
- detail_payload_bytes_mean: 53.0
- stage3_stream_bytes_mean: 175.0
- semantic_only_psnr_mean: 24.124753952026367
- semantic_only_l1_mean: 0.03963884711265564
- semantic_only_ms_ssim_mean: 0.8252045512199402
- stage3_psnr_mean: 24.44509506225586
- stage3_l1_mean: 0.037260085344314575
- stage3_ms_ssim_mean: 0.8271719813346863
- semantic_topk_hit_rate_mean: 0.75
- semantic_token_roundtrip_mean: 1.0
- detail_code_roundtrip_mean: 1.0
- residual_grid_abs_mean_mean: 0.013364642858505249
- residual_grid_std_mean: 0.01714456081390381
- residual_grid_clip_ratio_mean: 0.0
- detail_code_entropy_bits_mean: 2.1281793117523193
- stage3_psnr_delta_vs_semantic_only: 0.3203411102294922
- stage3_l1_delta_vs_semantic_only: -0.0023787617683410645
- stage3_ms_ssim_delta_vs_semantic_only: 0.0019674301147460938
- all_semantic_tokens_roundtrip: True
- all_detail_codes_roundtrip: True
- roundtrip_failure_count: 0
- roundtrip_failures: []
- per_image_metrics: results/bitstreams/stage3_uniform_residual/20260627_stage3_protocol_smoke_gencodec_div2kval1/per_image_metrics.jsonl

## Artifacts

- summary: `results/bitstreams/stage3_uniform_residual/20260627_stage3_protocol_smoke_gencodec_div2kval1/summary.json`
- per_image_metrics: `results/bitstreams/stage3_uniform_residual/20260627_stage3_protocol_smoke_gencodec_div2kval1/per_image_metrics.jsonl`
- sample: `results/bitstreams/stage3_uniform_residual/20260627_stage3_protocol_smoke_gencodec_div2kval1/stage3_uniform_residual_grid.png`
