# 20260627_stage1_rateprior0005_tau01_vq005_b16_coser_common_lic_perimage_recon

Date: 2026-06-27T23:37:28

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage1_semantic_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt --eval-protocol coser_common_lic --batch-size 8 --num-workers 4 --compute-perceptual --codec fixed_bits --write-per-image-metrics --save-reconstructions --wandb-mode offline --run-name 20260627_stage1_rateprior0005_tau01_vq005_b16_coser_common_lic_perimage_recon
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt
- num_images: 165
- crop_size: 256
- semantic_tokens_per_image: 64
- semantic_tokens_per_image_min: 64
- semantic_tokens_per_image_max: 64
- codebook_size: 8192
- main_bpp_metric: actual_payload_bpp_mean
- paper_bpp_metric: paper_bpp_mean
- debug_bpp_metric: debug_full_stream_bpp_mean
- eval_protocol: coser_common_lic
- eval_datasets: []
- eval_image_roots: ['/dpl/kodak', '/dpl/clic/professional/valid', '/dpl/div2k']
- eval_protocol_summary: {'name': 'coser_common_lic', 'display_name': 'CoSER common LIC protocol', 'dataset_keys': ['kodak', 'clic_professional_valid', 'div2k_val'], 'dataset_counts': {'kodak': 24, 'clic_professional_valid': 41, 'div2k_val': 100}, 'expected_counts': {'kodak': 24, 'clic_professional_valid': 41, 'div2k_val': 100}, 'count_status': {'kodak': 'ok', 'clic_professional_valid': 'ok', 'div2k_val': 'ok'}, 'total_images': 165, 'primary_metrics': ['PSNR', 'MS-SSIM', 'LPIPS', 'DISTS', 'FID/KID when sample handling is defined'], 'bpp_metric': 'actual_payload_bpp', 'default_crop_size': None, 'notes': ['Internal CoSER LIC protocol for Kodak 24, CLIC Professional Validation 41, and DIV2K validation 100.', 'Do not merge this table with CLIC2020 test 428 comparisons.'], 'datasets': [{'key': 'kodak', 'display_name': 'Kodak 24', 'count': 24, 'expected_count': 24, 'status': 'ok', 'source_roots': ['/dpl/kodak'], 'first_path': '/dpl/kodak/kodim01.png', 'last_path': '/dpl/kodak/kodim24.png', 'notes': []}, {'key': 'clic_professional_valid', 'display_name': 'CLIC Professional Validation 41', 'count': 41, 'expected_count': 41, 'status': 'ok', 'source_roots': ['/dpl/clic/professional/valid'], 'first_path': '/dpl/clic/professional/valid/alberto-montalesi-176097.png', 'last_path': '/dpl/clic/professional/valid/zugr-108.png', 'notes': []}, {'key': 'div2k_val', 'display_name': 'DIV2K validation 100', 'count': 100, 'expected_count': 100, 'status': 'ok', 'source_roots': ['/dpl/div2k'], 'first_path': '/dpl/div2k/0801.png', 'last_path': '/dpl/div2k/0900.png', 'notes': ['Resolved by filename indices 0801-0900 from a mixed DIV2K root.']}]}
- deterministic_eval: True
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
- fixed_bits_psnr_mean: 20.86491964513605
- fixed_bits_l1_mean: 0.06745416644509092
- fixed_bits_ms_ssim_mean: 0.6904037164919304
- fixed_bits_token_roundtrip_mean: 1.0
- fixed_bits_lpips_alex_mean: 0.6077934628860517
- fixed_bits_dists_mean: 0.38456118684826474
- fixed_bits_all_tokens_roundtrip: True

## Artifacts

- summary: `results/bitstreams/stage1_semantic_vq/20260627_stage1_rateprior0005_tau01_vq005_b16_coser_common_lic_perimage_recon/summary.json`
- output_dir: `results/bitstreams/stage1_semantic_vq/20260627_stage1_rateprior0005_tau01_vq005_b16_coser_common_lic_perimage_recon`
- streams: ``
- reconstructions: `results/bitstreams/stage1_semantic_vq/20260627_stage1_rateprior0005_tau01_vq005_b16_coser_common_lic_perimage_recon/reconstructions`
- per_image_metrics: `results/bitstreams/stage1_semantic_vq/20260627_stage1_rateprior0005_tau01_vq005_b16_coser_common_lic_perimage_recon/per_image_metrics.jsonl`
