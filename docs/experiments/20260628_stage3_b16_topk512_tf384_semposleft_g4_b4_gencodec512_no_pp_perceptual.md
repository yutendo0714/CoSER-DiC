# 20260628_stage3_b16_topk512_tf384_semposleft_g4_b4_gencodec512_no_pp_perceptual

Date: 2026-06-28T03:00:10

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage3_uniform_residual_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt --semantic-prior outputs/stage2_learned_entropy/20260628_stage2_learned_topk512_escape_huffman_fit_32768tokens_crop512_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json --detail-prior outputs/stage3_residual_entropy/20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_crop512_8192calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json --eval-protocol gencodec_reproduction --crop-size 512 --batch-size 1 --num-workers 4 --detail-downsample-factor 32 --detail-bits 4 --detail-range 0.25 --detail-codec semantic_position_leftctx_huffman --compute-perceptual --wandb-mode offline --run-name 20260628_stage3_b16_topk512_tf384_semposleft_g4_b4_gencodec512_no_pp_perceptual
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt
- semantic_prior: outputs/stage2_learned_entropy/20260628_stage2_learned_topk512_escape_huffman_fit_32768tokens_crop512_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json
- detail_prior: outputs/stage3_residual_entropy/20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_crop512_8192calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260628_stage2_token_prior_crop512_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_amp_es.pt
- num_images: 552
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
- compute_perceptual: True
- save_reconstructions: False
- save_reconstruction_limit: 0
- save_reconstruction_triptychs: False
- residual_code: {'codec': 'static_residual_grid_semantic_position_leftctx_huffman', 'version': 0, 'bits': 4, 'value_range': 0.25, 'payload_codec': 'semantic_position_leftctx_huffman', 'detail_shape': [3, 16, 16], 'semantic_shape': [16, 16], 'group_count': 4, 'left_context_mode': 'start_left_sign4', 'left_context_count': 4, 'num_position_group_context_codes': 12288, 'min_code_length': 1, 'max_code_length': 14, 'mean_code_length_unweighted': 6.5782216389973955}
- semantic_payload_bpp_mean: 0.009223329848137455
- detail_payload_bpp_mean: 0.005110588626585145
- actual_payload_bpp_mean: 0.014333918474722599
- paper_bpp_mean: 0.014333918474722599
- total_payload_bpp_mean: 0.014333918474722599
- debug_semantic_only_full_stream_bpp_mean: 0.023688661879387455
- debug_full_stream_bpp_mean: 0.0292264965997226
- semantic_only_full_stream_bpp_mean: 0.023688661879387455
- stage3_full_stream_bpp_mean: 0.0292264965997226
- semantic_payload_bytes_mean: 302.2300724637681
- detail_payload_bytes_mean: 167.46376811594203
- stage3_stream_bytes_mean: 957.6938405797101
- semantic_only_psnr_mean: 21.445535329804905
- semantic_only_l1_mean: 0.05927534353500907
- semantic_only_ms_ssim_mean: 0.7268036636186467
- stage3_psnr_mean: 21.92317509651184
- stage3_l1_mean: 0.05648139264076894
- stage3_ms_ssim_mean: 0.7324484130907534
- semantic_topk_hit_rate_mean: 0.7686112998188406
- semantic_token_roundtrip_mean: 0.9818840579710145
- detail_code_roundtrip_mean: 1.0
- residual_grid_abs_mean_mean: 0.020784146367552003
- residual_grid_std_mean: 0.026560631646450773
- residual_grid_clip_ratio_mean: 0.0016535515439328806
- detail_code_entropy_bits_mean: 1.6890001522674076
- semantic_only_lpips_alex_mean: 0.5802656617690471
- semantic_only_dists_mean: 0.35890114944482193
- stage3_lpips_alex_mean: 0.5809743574456029
- stage3_dists_mean: 0.35553381343682605
- stage3_psnr_delta_vs_semantic_only: 0.4776397667069361
- stage3_l1_delta_vs_semantic_only: -0.0027939508942401298
- stage3_ms_ssim_delta_vs_semantic_only: 0.0056447494721066915
- stage3_lpips_alex_delta_vs_semantic_only: 0.0007086956765557373
- stage3_dists_delta_vs_semantic_only: -0.0033673360079958803
- all_semantic_tokens_roundtrip: False
- all_detail_codes_roundtrip: True
- roundtrip_failure_count: 10
- roundtrip_failures: [{'index': 8, 'path': '/dpl/kodak/kodim09.png', 'semantic_token_roundtrip': False, 'stream_semantic_roundtrip': False, 'detail_code_roundtrip': True, 'stream_detail_roundtrip': True}, {'index': 20, 'path': '/dpl/kodak/kodim21.png', 'semantic_token_roundtrip': False, 'stream_semantic_roundtrip': False, 'detail_code_roundtrip': True, 'stream_detail_roundtrip': True}, {'index': 35, 'path': '/dpl/clic/professional/test/0d57c99aa83305d93d8fc5d21910343a.png', 'semantic_token_roundtrip': False, 'stream_semantic_roundtrip': False, 'detail_code_roundtrip': True, 'stream_detail_roundtrip': True}, {'index': 40, 'path': '/dpl/clic/professional/test/12e237070b800d8d4602fb0892591464.png', 'semantic_token_roundtrip': False, 'stream_semantic_roundtrip': False, 'detail_code_roundtrip': True, 'stream_detail_roundtrip': True}, {'index': 119, 'path': '/dpl/clic/professional/test/61b8f705d907b64bc4be9b69c25c1fc6.png', 'semantic_token_roundtrip': False, 'stream_semantic_roundtrip': False, 'detail_code_roundtrip': True, 'stream_detail_roundtrip': True}, {'index': 181, 'path': '/dpl/clic/professional/test/a8a237b591fe13328c534c388b0e74cb.png', 'semantic_token_roundtrip': False, 'stream_semantic_roundtrip': False, 'detail_code_roundtrip': True, 'stream_detail_roundtrip': True}, {'index': 203, 'path': '/dpl/clic/professional/test/b9bad0c68eb9ce94e02e9698c8cc429a.png', 'semantic_token_roundtrip': False, 'stream_semantic_roundtrip': False, 'detail_code_roundtrip': True, 'stream_detail_roundtrip': True}, {'index': 312, 'path': '/dpl/clic/mobile/test/2a4a019c669ffec1ff91b094c627adaf.png', 'semantic_token_roundtrip': False, 'stream_semantic_roundtrip': False, 'detail_code_roundtrip': True, 'stream_detail_roundtrip': True}, {'index': 407, 'path': '/dpl/clic/mobile/test/bf1a05045600a8237ba5a5beaade0d02.png', 'semantic_token_roundtrip': False, 'stream_semantic_roundtrip': False, 'detail_code_roundtrip': True, 'stream_detail_roundtrip': True}, {'index': 546, 'path': '/dpl/div2k/0895.png', 'semantic_token_roundtrip': False, 'stream_semantic_roundtrip': False, 'detail_code_roundtrip': True, 'stream_detail_roundtrip': True}]
- per_image_metrics: results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk512_tf384_semposleft_g4_b4_gencodec512_no_pp_perceptual/per_image_metrics.jsonl

## Artifacts

- summary: `results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk512_tf384_semposleft_g4_b4_gencodec512_no_pp_perceptual/summary.json`
- per_image_metrics: `results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk512_tf384_semposleft_g4_b4_gencodec512_no_pp_perceptual/per_image_metrics.jsonl`
- sample: `results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk512_tf384_semposleft_g4_b4_gencodec512_no_pp_perceptual/stage3_uniform_residual_grid.png`
