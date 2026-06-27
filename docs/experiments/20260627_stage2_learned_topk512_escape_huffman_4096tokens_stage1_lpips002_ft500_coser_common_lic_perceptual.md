# 20260627_stage2_learned_topk512_escape_huffman_4096tokens_stage1_lpips002_ft500_coser_common_lic_perceptual

Date: 2026-06-27T22:41:20

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage2_learned_topk_escape_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_active5k_lpips002_fp32loss_ft500.pt --prior outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_4096tokens_stage1_lpips002_ft500_lowlr3k/learned_topk_escape_huffman_prior.json --token-prior-checkpoint checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_3kstep_4096tokens_stage1_lpips002_ft500_lr1e4_do02_es.pt --eval-protocol coser_common_lic --batch-size 4 --num-workers 4 --stream-header-codec compact --stream-checksum-codec crc32 --compute-perceptual --wandb-mode offline --run-name 20260627_stage2_learned_topk512_escape_huffman_4096tokens_stage1_lpips002_ft500_coser_common_lic_perceptual
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_active5k_lpips002_fp32loss_ft500.pt
- prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_4096tokens_stage1_lpips002_ft500_lowlr3k/learned_topk_escape_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_3kstep_4096tokens_stage1_lpips002_ft500_lr1e4_do02_es.pt
- num_images: 165
- crop_size: 256
- codebook_size: 8192
- topk: 512
- stream_header_codec: compact
- stream_checksum_codec: crc32
- main_bpp_metric: learned_topk_escape_huffman_actual_payload_bpp_mean
- paper_bpp_metric: learned_topk_escape_huffman_paper_bpp_mean
- debug_bpp_metric: learned_topk_escape_huffman_debug_full_stream_bpp_mean
- eval_protocol: coser_common_lic
- eval_datasets: []
- eval_image_roots: ['/dpl/kodak', '/dpl/clic/professional/valid', '/dpl/div2k']
- eval_protocol_summary: {'name': 'coser_common_lic', 'display_name': 'CoSER common LIC protocol', 'dataset_keys': ['kodak', 'clic_professional_valid', 'div2k_val'], 'dataset_counts': {'kodak': 24, 'clic_professional_valid': 41, 'div2k_val': 100}, 'expected_counts': {'kodak': 24, 'clic_professional_valid': 41, 'div2k_val': 100}, 'count_status': {'kodak': 'ok', 'clic_professional_valid': 'ok', 'div2k_val': 'ok'}, 'total_images': 165, 'primary_metrics': ['PSNR', 'MS-SSIM', 'LPIPS', 'DISTS', 'FID/KID when sample handling is defined'], 'bpp_metric': 'actual_payload_bpp', 'default_crop_size': None, 'notes': ['Internal CoSER LIC protocol for Kodak 24, CLIC Professional Validation 41, and DIV2K validation 100.', 'Do not merge this table with CLIC2020 test 428 comparisons.'], 'datasets': [{'key': 'kodak', 'display_name': 'Kodak 24', 'count': 24, 'expected_count': 24, 'status': 'ok', 'source_roots': ['/dpl/kodak'], 'first_path': '/dpl/kodak/kodim01.png', 'last_path': '/dpl/kodak/kodim24.png', 'notes': []}, {'key': 'clic_professional_valid', 'display_name': 'CLIC Professional Validation 41', 'count': 41, 'expected_count': 41, 'status': 'ok', 'source_roots': ['/dpl/clic/professional/valid'], 'first_path': '/dpl/clic/professional/valid/alberto-montalesi-176097.png', 'last_path': '/dpl/clic/professional/valid/zugr-108.png', 'notes': []}, {'key': 'div2k_val', 'display_name': 'DIV2K validation 100', 'count': 100, 'expected_count': 100, 'status': 'ok', 'source_roots': ['/dpl/div2k'], 'first_path': '/dpl/div2k/0801.png', 'last_path': '/dpl/div2k/0900.png', 'notes': ['Resolved by filename indices 0801-0900 from a mixed DIV2K root.']}]}
- deterministic_eval: True
- compute_perceptual: True
- learned_topk_escape_huffman_payload_bpp_mean: 0.012452651515151516
- learned_topk_escape_huffman_actual_payload_bpp_mean: 0.012452651515151516
- learned_topk_escape_huffman_paper_bpp_mean: 0.012452651515151516
- learned_topk_escape_huffman_actual_bpp_mean: 0.017213393702651514
- learned_topk_escape_huffman_debug_full_stream_bpp_mean: 0.017213393702651514
- learned_topk_escape_huffman_stream_bytes_mean: 141.0121212121212
- learned_topk_escape_huffman_payload_bytes_mean: 102.01212121212122
- learned_topk_escape_huffman_psnr_mean: 20.836226937265106
- learned_topk_escape_huffman_l1_mean: 0.06787524664718093
- learned_topk_escape_huffman_ms_ssim_mean: 0.6905045213121357
- learned_topk_escape_huffman_roundtrip_mean: 1.0
- learned_topk_escape_huffman_topk_hit_rate_mean: 0.3731060606060606
- learned_topk_escape_huffman_unpadded_bits_per_token_mean: 12.698011363636363
- learned_topk_escape_huffman_lpips_alex_mean: 0.6275915798702927
- learned_topk_escape_huffman_dists_mean: 0.3919758865327546
- learned_topk_escape_huffman_all_tokens_roundtrip: True
- learned_topk_escape_huffman_roundtrip_failure_count: 0
- fixed_bits_payload_bpp_mean: 0.0126953125
- fixed_bits_actual_payload_bpp_mean: 0.0126953125
- fixed_bits_paper_bpp_mean: 0.0126953125
- fixed_bits_actual_bpp_mean: 0.0174560546875
- fixed_bits_debug_full_stream_bpp_mean: 0.0174560546875
- fixed_bits_stream_bytes_mean: 143.0
- fixed_bits_payload_bytes_mean: 104.0
- fixed_bits_psnr_mean: 20.836226937265106
- fixed_bits_l1_mean: 0.06787524664718093
- fixed_bits_ms_ssim_mean: 0.6905045213121357
- fixed_bits_roundtrip_mean: 1.0
- fixed_bits_lpips_alex_mean: 0.6275915798702927
- fixed_bits_dists_mean: 0.3919758865327546
- fixed_bits_all_tokens_roundtrip: True
- fixed_bits_roundtrip_failure_count: 0
- learned_topk_escape_huffman_payload_bpp_delta_vs_fixed_bits: -0.00024266098484848446
- learned_topk_escape_huffman_payload_bpp_delta_vs_leftctx_static_huffman: 0.0007237289891098495
- roundtrip_failures: []

## Artifacts

- summary: `results/bitstreams/stage2_learned_topk_escape/20260627_stage2_learned_topk512_escape_huffman_4096tokens_stage1_lpips002_ft500_coser_common_lic_perceptual/summary.json`
- sample: `results/bitstreams/stage2_learned_topk_escape/20260627_stage2_learned_topk512_escape_huffman_4096tokens_stage1_lpips002_ft500_coser_common_lic_perceptual/stage2_learned_topk_escape_recon_grid.png`
