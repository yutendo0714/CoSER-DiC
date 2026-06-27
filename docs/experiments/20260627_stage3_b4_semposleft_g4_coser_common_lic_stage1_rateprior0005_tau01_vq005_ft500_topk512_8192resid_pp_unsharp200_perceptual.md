# 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_rateprior0005_tau01_vq005_ft500_topk512_8192resid_pp_unsharp200_perceptual

Date: 2026-06-27T23:14:27

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage3_uniform_residual_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500.pt --semantic-prior outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_4096tokens_stage1_rateprior0005_tau01_vq005_ft500_lowlr3k/learned_topk_escape_huffman_prior.json --token-prior-checkpoint checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_3kstep_4096tokens_stage1_rateprior0005_tau01_vq005_ft500_lr1e4_do02_es.pt --detail-prior outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_stage1_rateprior0005_tau01_vq005_ft500/static_residual_grid_semantic_position_leftctx_huffman_prior.json --eval-protocol coser_common_lic --batch-size 4 --num-workers 4 --detail-codec semantic_position_leftctx_huffman --detail-downsample-factor 32 --detail-bits 4 --detail-range 0.25 --stream-header-codec compact --stream-checksum-codec crc32 --compute-perceptual --decoder-postprocess unsharp3x3 --decoder-postprocess-strength 2.0 --wandb-mode offline --run-name 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_rateprior0005_tau01_vq005_ft500_topk512_8192resid_pp_unsharp200_perceptual
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500.pt
- semantic_prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_4096tokens_stage1_rateprior0005_tau01_vq005_ft500_lowlr3k/learned_topk_escape_huffman_prior.json
- detail_prior: outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_stage1_rateprior0005_tau01_vq005_ft500/static_residual_grid_semantic_position_leftctx_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_3kstep_4096tokens_stage1_rateprior0005_tau01_vq005_ft500_lr1e4_do02_es.pt
- num_images: 165
- crop_size: 256
- semantic_topk: 512
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
- residual_code: {'codec': 'static_residual_grid_semantic_position_leftctx_huffman', 'version': 0, 'bits': 4, 'value_range': 0.25, 'payload_codec': 'semantic_position_leftctx_huffman', 'detail_shape': [3, 8, 8], 'semantic_shape': [8, 8], 'group_count': 4, 'left_context_mode': 'start_left_sign4', 'left_context_count': 4, 'num_position_group_context_codes': 3072, 'min_code_length': 1, 'max_code_length': 14, 'mean_code_length_unweighted': 6.471028645833333}
- semantic_payload_bpp_mean: 0.012369791666666666
- detail_payload_bpp_mean: 0.005222389914772727
- actual_payload_bpp_mean: 0.017592181581439395
- paper_bpp_mean: 0.017592181581439395
- total_payload_bpp_mean: 0.017592181581439395
- debug_semantic_only_full_stream_bpp_mean: 0.017130533854166668
- debug_full_stream_bpp_mean: 0.026869525331439395
- semantic_only_full_stream_bpp_mean: 0.017130533854166668
- stage3_full_stream_bpp_mean: 0.026869525331439395
- semantic_payload_bytes_mean: 101.33333333333333
- detail_payload_bytes_mean: 42.78181818181818
- stage3_stream_bytes_mean: 220.11515151515152
- semantic_only_psnr_mean: 20.8588397286155
- semantic_only_l1_mean: 0.06748685339522181
- semantic_only_ms_ssim_mean: 0.6911363135684621
- stage3_psnr_mean: 21.187439467690208
- stage3_l1_mean: 0.06550027286408074
- stage3_ms_ssim_mean: 0.694023428180001
- semantic_topk_hit_rate_mean: 0.3860795454545455
- semantic_token_roundtrip_mean: 1.0
- detail_code_roundtrip_mean: 1.0
- residual_grid_abs_mean_mean: 0.021178323792463
- residual_grid_std_mean: 0.02560577660121701
- residual_grid_clip_ratio_mean: 0.0
- detail_code_entropy_bits_mean: 1.6855884911887573
- semantic_only_lpips_alex_mean: 0.6151886095377532
- semantic_only_dists_mean: 0.3871394182696487
- stage3_lpips_alex_mean: 0.5847654103103912
- stage3_dists_mean: 0.37227355133403434
- stage3_psnr_delta_vs_semantic_only: 0.32859973907470774
- stage3_l1_delta_vs_semantic_only: -0.0019865805311410772
- stage3_ms_ssim_delta_vs_semantic_only: 0.002887114611538899
- stage3_lpips_alex_delta_vs_semantic_only: -0.03042319922736203
- stage3_dists_delta_vs_semantic_only: -0.014865866935614369
- all_semantic_tokens_roundtrip: True
- all_detail_codes_roundtrip: True
- roundtrip_failure_count: 0
- roundtrip_failures: []
- per_image_metrics: results/bitstreams/stage3_uniform_residual/20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_rateprior0005_tau01_vq005_ft500_topk512_8192resid_pp_unsharp200_perceptual/per_image_metrics.jsonl

## Artifacts

- summary: `results/bitstreams/stage3_uniform_residual/20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_rateprior0005_tau01_vq005_ft500_topk512_8192resid_pp_unsharp200_perceptual/summary.json`
- per_image_metrics: `results/bitstreams/stage3_uniform_residual/20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_rateprior0005_tau01_vq005_ft500_topk512_8192resid_pp_unsharp200_perceptual/per_image_metrics.jsonl`
- sample: `results/bitstreams/stage3_uniform_residual/20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_rateprior0005_tau01_vq005_ft500_topk512_8192resid_pp_unsharp200_perceptual/stage3_uniform_residual_grid.png`
