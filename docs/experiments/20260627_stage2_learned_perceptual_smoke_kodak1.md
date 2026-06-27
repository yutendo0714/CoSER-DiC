# 20260627_stage2_learned_perceptual_smoke_kodak1

Date: 2026-06-27T16:40:21

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage2_learned_topk_escape_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --prior outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json --token-prior-checkpoint checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt --eval-protocol coser_common_lic --eval-dataset kodak --max-images 1 --crop-size 256 --batch-size 1 --num-workers 0 --stream-header-codec compact --stream-checksum-codec crc32 --compute-perceptual --wandb-mode offline --run-name 20260627_stage2_learned_perceptual_smoke_kodak1
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt
- num_images: 1
- crop_size: 256
- codebook_size: 8192
- topk: 512
- stream_header_codec: compact
- stream_checksum_codec: crc32
- main_bpp_metric: learned_topk_escape_huffman_actual_payload_bpp_mean
- paper_bpp_metric: learned_topk_escape_huffman_paper_bpp_mean
- debug_bpp_metric: learned_topk_escape_huffman_debug_full_stream_bpp_mean
- eval_protocol: coser_common_lic
- eval_datasets: ['kodak']
- eval_image_roots: ['/dpl/kodak']
- eval_protocol_summary: {'name': 'coser_common_lic', 'display_name': 'CoSER common LIC protocol', 'dataset_keys': ['kodak'], 'dataset_counts': {'kodak': 24}, 'expected_counts': {'kodak': 24}, 'count_status': {'kodak': 'ok'}, 'total_images': 24, 'primary_metrics': ['PSNR', 'MS-SSIM', 'LPIPS', 'DISTS', 'FID/KID when sample handling is defined'], 'bpp_metric': 'actual_payload_bpp', 'default_crop_size': None, 'notes': ['Internal CoSER LIC protocol for Kodak 24, CLIC Professional Validation 41, and DIV2K validation 100.', 'Do not merge this table with CLIC2020 test 428 comparisons.'], 'datasets': [{'key': 'kodak', 'display_name': 'Kodak 24', 'count': 24, 'expected_count': 24, 'status': 'ok', 'source_roots': ['/dpl/kodak'], 'first_path': '/dpl/kodak/kodim01.png', 'last_path': '/dpl/kodak/kodim24.png', 'notes': []}]}
- compute_perceptual: True
- learned_topk_escape_huffman_payload_bpp_mean: 0.01123046875
- learned_topk_escape_huffman_actual_payload_bpp_mean: 0.01123046875
- learned_topk_escape_huffman_paper_bpp_mean: 0.01123046875
- learned_topk_escape_huffman_actual_bpp_mean: 0.0159912109375
- learned_topk_escape_huffman_debug_full_stream_bpp_mean: 0.0159912109375
- learned_topk_escape_huffman_stream_bytes_mean: 131.0
- learned_topk_escape_huffman_payload_bytes_mean: 92.0
- learned_topk_escape_huffman_psnr_mean: 19.02641487121582
- learned_topk_escape_huffman_l1_mean: 0.08127432316541672
- learned_topk_escape_huffman_ms_ssim_mean: 0.5197452902793884
- learned_topk_escape_huffman_roundtrip_mean: 1.0
- learned_topk_escape_huffman_topk_hit_rate_mean: 0.625
- learned_topk_escape_huffman_unpadded_bits_per_token_mean: 11.484375
- learned_topk_escape_huffman_lpips_alex_mean: 0.8745201826095581
- learned_topk_escape_huffman_dists_mean: 0.4744288921356201
- learned_topk_escape_huffman_all_tokens_roundtrip: True
- learned_topk_escape_huffman_roundtrip_failure_count: 0
- fixed_bits_payload_bpp_mean: 0.0126953125
- fixed_bits_actual_payload_bpp_mean: 0.0126953125
- fixed_bits_paper_bpp_mean: 0.0126953125
- fixed_bits_actual_bpp_mean: 0.0174560546875
- fixed_bits_debug_full_stream_bpp_mean: 0.0174560546875
- fixed_bits_stream_bytes_mean: 143.0
- fixed_bits_payload_bytes_mean: 104.0
- fixed_bits_psnr_mean: 19.02641487121582
- fixed_bits_l1_mean: 0.08127432316541672
- fixed_bits_ms_ssim_mean: 0.5197452902793884
- fixed_bits_roundtrip_mean: 1.0
- fixed_bits_lpips_alex_mean: 0.8745201826095581
- fixed_bits_dists_mean: 0.4744288921356201
- fixed_bits_all_tokens_roundtrip: True
- fixed_bits_roundtrip_failure_count: 0
- learned_topk_escape_huffman_payload_bpp_delta_vs_fixed_bits: -0.00146484375
- learned_topk_escape_huffman_payload_bpp_delta_vs_leftctx_static_huffman: -0.0004984537760416661
- roundtrip_failures: []

## Artifacts

- summary: `results/bitstreams/stage2_learned_topk_escape/20260627_stage2_learned_perceptual_smoke_kodak1/summary.json`
- sample: `results/bitstreams/stage2_learned_topk_escape/20260627_stage2_learned_perceptual_smoke_kodak1/stage2_learned_topk_escape_recon_grid.png`
