# 20260627_stage1_perceptual_smoke_kodak1

Date: 2026-06-27T16:28:39

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage1_semantic_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --eval-protocol coser_common_lic --eval-dataset kodak --max-images 1 --crop-size 256 --batch-size 1 --num-workers 0 --codec fixed_bits --compute-perceptual --wandb-mode offline --run-name 20260627_stage1_perceptual_smoke_kodak1
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- num_images: 1
- crop_size: 256
- semantic_tokens_per_image: 64
- semantic_tokens_per_image_min: 64
- semantic_tokens_per_image_max: 64
- codebook_size: 8192
- main_bpp_metric: actual_payload_bpp_mean
- paper_bpp_metric: paper_bpp_mean
- debug_bpp_metric: debug_full_stream_bpp_mean
- eval_protocol: coser_common_lic
- eval_datasets: ['kodak']
- eval_image_roots: ['/dpl/kodak']
- eval_protocol_summary: {'name': 'coser_common_lic', 'display_name': 'CoSER common LIC protocol', 'dataset_keys': ['kodak'], 'dataset_counts': {'kodak': 24}, 'expected_counts': {'kodak': 24}, 'count_status': {'kodak': 'ok'}, 'total_images': 24, 'primary_metrics': ['PSNR', 'MS-SSIM', 'LPIPS', 'DISTS', 'FID/KID when sample handling is defined'], 'bpp_metric': 'actual_payload_bpp', 'default_crop_size': None, 'notes': ['Internal CoSER LIC protocol for Kodak 24, CLIC Professional Validation 41, and DIV2K validation 100.', 'Do not merge this table with CLIC2020 test 428 comparisons.'], 'datasets': [{'key': 'kodak', 'display_name': 'Kodak 24', 'count': 24, 'expected_count': 24, 'status': 'ok', 'source_roots': ['/dpl/kodak'], 'first_path': '/dpl/kodak/kodim01.png', 'last_path': '/dpl/kodak/kodim24.png', 'notes': []}]}
- compute_perceptual: True
- fixed_bits_actual_bpp_mean: 0.0743408203125
- fixed_bits_actual_payload_bpp_mean: 0.0126953125
- fixed_bits_paper_bpp_mean: 0.0126953125
- fixed_bits_debug_full_stream_bpp_mean: 0.0743408203125
- fixed_bits_token_payload_bpp_mean: 0.0126953125
- fixed_bits_container_overhead_bpp_mean: 0.0616455078125
- fixed_bits_stream_bytes_mean: 609.0
- fixed_bits_token_payload_bytes_mean: 104.0
- fixed_bits_container_overhead_bytes_mean: 505.0
- fixed_bits_psnr_mean: 19.02641487121582
- fixed_bits_l1_mean: 0.08127432316541672
- fixed_bits_ms_ssim_mean: 0.5197452902793884
- fixed_bits_token_roundtrip_mean: 1.0
- fixed_bits_lpips_alex_mean: 0.8745201826095581
- fixed_bits_dists_mean: 0.4744294285774231
- fixed_bits_all_tokens_roundtrip: True

## Artifacts

- summary: `results/bitstreams/stage1_semantic_vq/20260627_stage1_perceptual_smoke_kodak1/summary.json`
- output_dir: `results/bitstreams/stage1_semantic_vq/20260627_stage1_perceptual_smoke_kodak1`
- streams: ``
