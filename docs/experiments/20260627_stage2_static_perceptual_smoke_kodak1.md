# 20260627_stage2_static_perceptual_smoke_kodak1

Date: 2026-06-27T16:40:22

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage2_static_huffman_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --prior outputs/stage2_semantic_entropy/20260627_stage2_leftctx_huffman_top64_b4096_4096calib_oi_div2k_from_stage1_best/static_left_context_huffman_prior.json --eval-protocol coser_common_lic --eval-dataset kodak --max-images 1 --crop-size 256 --batch-size 1 --num-workers 0 --stream-header-codec compact --stream-checksum-codec crc32 --compute-perceptual --wandb-mode offline --run-name 20260627_stage2_static_perceptual_smoke_kodak1
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- prior: outputs/stage2_semantic_entropy/20260627_stage2_leftctx_huffman_top64_b4096_4096calib_oi_div2k_from_stage1_best/static_left_context_huffman_prior.json
- num_images: 1
- crop_size: 256
- codebook_size: 8192
- stream_header_codec: compact
- stream_checksum_codec: crc32
- main_bpp_metric: static_left_context_huffman_actual_payload_bpp_mean
- paper_bpp_metric: static_left_context_huffman_paper_bpp_mean
- debug_bpp_metric: static_left_context_huffman_debug_full_stream_bpp_mean
- eval_protocol: coser_common_lic
- eval_datasets: ['kodak']
- eval_image_roots: ['/dpl/kodak']
- eval_protocol_summary: {'name': 'coser_common_lic', 'display_name': 'CoSER common LIC protocol', 'dataset_keys': ['kodak'], 'dataset_counts': {'kodak': 24}, 'expected_counts': {'kodak': 24}, 'count_status': {'kodak': 'ok'}, 'total_images': 24, 'primary_metrics': ['PSNR', 'MS-SSIM', 'LPIPS', 'DISTS', 'FID/KID when sample handling is defined'], 'bpp_metric': 'actual_payload_bpp', 'default_crop_size': None, 'notes': ['Internal CoSER LIC protocol for Kodak 24, CLIC Professional Validation 41, and DIV2K validation 100.', 'Do not merge this table with CLIC2020 test 428 comparisons.'], 'datasets': [{'key': 'kodak', 'display_name': 'Kodak 24', 'count': 24, 'expected_count': 24, 'status': 'ok', 'source_roots': ['/dpl/kodak'], 'first_path': '/dpl/kodak/kodim01.png', 'last_path': '/dpl/kodak/kodim24.png', 'notes': []}]}
- compute_perceptual: True
- static_left_context_huffman_payload_bpp_mean: 0.0120849609375
- static_left_context_huffman_actual_payload_bpp_mean: 0.0120849609375
- static_left_context_huffman_paper_bpp_mean: 0.0120849609375
- static_left_context_huffman_actual_bpp_mean: 0.0181884765625
- static_left_context_huffman_debug_full_stream_bpp_mean: 0.0181884765625
- static_left_context_huffman_stream_bytes_mean: 149.0
- static_left_context_huffman_payload_bytes_mean: 99.0
- static_left_context_huffman_psnr_mean: 19.026418685913086
- static_left_context_huffman_l1_mean: 0.0812743753194809
- static_left_context_huffman_ms_ssim_mean: 0.5197446942329407
- static_left_context_huffman_roundtrip_mean: 1.0
- static_left_context_huffman_lpips_alex_mean: 0.8745080232620239
- static_left_context_huffman_dists_mean: 0.4744170904159546
- static_left_context_huffman_all_tokens_roundtrip: True
- fixed_bits_payload_bpp_mean: 0.0126953125
- fixed_bits_actual_payload_bpp_mean: 0.0126953125
- fixed_bits_paper_bpp_mean: 0.0126953125
- fixed_bits_actual_bpp_mean: 0.0174560546875
- fixed_bits_debug_full_stream_bpp_mean: 0.0174560546875
- fixed_bits_stream_bytes_mean: 143.0
- fixed_bits_payload_bytes_mean: 104.0
- fixed_bits_psnr_mean: 19.026418685913086
- fixed_bits_l1_mean: 0.0812743753194809
- fixed_bits_ms_ssim_mean: 0.5197446942329407
- fixed_bits_roundtrip_mean: 1.0
- fixed_bits_lpips_alex_mean: 0.8745080232620239
- fixed_bits_dists_mean: 0.4744170904159546
- fixed_bits_all_tokens_roundtrip: True
- static_left_context_huffman_payload_bpp_delta_vs_fixed_bits: -0.0006103515625

## Artifacts

- summary: `results/bitstreams/stage2_static_huffman/20260627_stage2_static_perceptual_smoke_kodak1/summary.json`
- output_dir: `results/bitstreams/stage2_static_huffman/20260627_stage2_static_perceptual_smoke_kodak1`
- streams: ``
