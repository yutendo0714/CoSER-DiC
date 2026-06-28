# 20260628_stage4_decoder_refiner_ckpt250sweep_step500_cod512_full552_eval

Date: 2026-06-28T14:24:51

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage3_uniform_residual_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt --semantic-prior outputs/stage2_learned_entropy/20260628_stage2_learned_topk2048_escape_huffman_fit_32768tokens_crop512_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json --token-prior-checkpoint checkpoints/stage2_token_prior/20260628_stage2_token_prior_crop512_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_amp_es.pt --detail-codec semantic_position_leftctx_huffman --detail-prior outputs/stage3_residual_entropy/20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_crop512_8192calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json --eval-protocol cod_reproduction_512 --crop-size 512 --batch-size 1 --detail-downsample-factor 32 --detail-bits 4 --detail-range 0.25 --detail-gain 0.5 --decoder-postprocess unsharp3x3 --decoder-postprocess-strength 0.15 --decoder-refiner-checkpoint checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_ckpt250_1k_bs4_anchor025_rs005_tv005_chroma010_oi_only_step000500.pt --compute-perceptual --save-reconstructions --save-reconstruction-limit 552 --run-name 20260628_stage4_decoder_refiner_ckpt250sweep_step500_cod512_full552_eval --wandb-mode offline
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt
- semantic_prior: outputs/stage2_learned_entropy/20260628_stage2_learned_topk2048_escape_huffman_fit_32768tokens_crop512_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json
- detail_prior: outputs/stage3_residual_entropy/20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_crop512_8192calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260628_stage2_token_prior_crop512_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_amp_es.pt
- num_images: 552
- crop_size: 512
- protocol_default_crop_size: 512
- crop_size_matches_protocol_default: True
- semantic_topk: 2048
- semantic_topk_schedule: prefix_replay_decoder_safe
- detail_downsample_factor: 32
- detail_shape: [3, 16, 16]
- detail_bits: 4
- detail_range: 0.25
- detail_gain: 0.5
- decoder_postprocess: unsharp3x3
- decoder_postprocess_strength: 0.15
- decoder_refiner_checkpoint: checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_ckpt250_1k_bs4_anchor025_rs005_tv005_chroma010_oi_only_step000500.pt
- decoder_refiner_enabled: True
- decoder_refiner_config: {'image_channels': 3, 'semantic_channels': 256, 'detail_channels': 3, 'base_channels': 64, 'semantic_context_channels': 32, 'num_res_blocks': 6, 'residual_scale': 0.05, 'use_semantic_latent': True, 'zero_init_output': True}
- decoder_refiner_payload_policy: fixed decoder-side weights; no additional actual_payload_bpp
- detail_codec: semantic_position_leftctx_huffman
- stream_header_codec: json
- stream_checksum_codec: sha256
- main_bpp_metric: actual_payload_bpp_mean
- paper_bpp_metric: paper_bpp_mean
- debug_bpp_metric: debug_full_stream_bpp_mean
- eval_protocol: cod_reproduction_512
- eval_datasets: []
- eval_image_roots: ['/dpl/kodak', '/dpl/clic/professional/test', '/dpl/clic/mobile/test', '/dpl/div2k']
- eval_protocol_summary: {'name': 'cod_reproduction_512', 'display_name': 'CoD paper reproduction protocol, 512x512', 'dataset_keys': ['kodak', 'clic2020_test', 'div2k_val'], 'dataset_counts': {'kodak': 24, 'clic2020_test': 428, 'div2k_val': 100}, 'expected_counts': {'kodak': 24, 'clic2020_test': 428, 'div2k_val': 100}, 'count_status': {'kodak': 'ok', 'clic2020_test': 'ok', 'div2k_val': 'ok'}, 'total_images': 552, 'primary_metrics': ['PSNR', 'LPIPS', 'DISTS', 'FID'], 'bpp_metric': 'actual_payload_bpp for CoSER; official nominal/file bpp must be labeled for external codecs', 'default_crop_size': 512, 'notes': ['Primary CoSER low-bitrate generative comparison protocol for CoD-style tables.', 'Resize if needed and center-crop every image to 512x512 before evaluation.', 'Use CLIC2020 test 428 as professional/test plus mobile/test.', 'Use DIV2K validation images 0801-0900, not the first 100 files from a mixed DIV2K root.', 'For CoD paper patch FID, run per dataset and label exact settings: Kodak512 uses 64px patches with split=2; CLIC2020 uses 128px patches with split=2.', 'DIV2K patch FID should be reported only when the chosen patch size/backend is explicitly labeled.', 'Patch-based or overlapped-patch FID must label patch size, shift/split count, and backend.'], 'datasets': [{'key': 'kodak', 'display_name': 'Kodak 24', 'count': 24, 'expected_count': 24, 'status': 'ok', 'source_roots': ['/dpl/kodak'], 'first_path': '/dpl/kodak/kodim01.png', 'last_path': '/dpl/kodak/kodim24.png', 'notes': []}, {'key': 'clic2020_test', 'display_name': 'CLIC2020 test 428', 'count': 428, 'expected_count': 428, 'status': 'ok', 'source_roots': ['/dpl/clic/professional/test', '/dpl/clic/mobile/test'], 'first_path': '/dpl/clic/professional/test/01443654fdc57490d5c2d8b0fd9e065b.png', 'last_path': '/dpl/clic/mobile/test/ff52b7548f7024b96d2cea64e5e061b7.png', 'notes': ['professional/test and mobile/test are concatenated in that order.']}, {'key': 'div2k_val', 'display_name': 'DIV2K validation 100', 'count': 100, 'expected_count': 100, 'status': 'ok', 'source_roots': ['/dpl/div2k'], 'first_path': '/dpl/div2k/0801.png', 'last_path': '/dpl/div2k/0900.png', 'notes': ['Resolved by filename indices 0801-0900 from a mixed DIV2K root.']}]}
- deterministic_eval: True
- compute_perceptual: True
- save_reconstructions: True
- save_reconstruction_limit: 552
- save_reconstruction_triptychs: False
- residual_code: {'codec': 'static_residual_grid_semantic_position_leftctx_huffman', 'version': 0, 'bits': 4, 'value_range': 0.25, 'payload_codec': 'semantic_position_leftctx_huffman', 'detail_shape': [3, 16, 16], 'semantic_shape': [16, 16], 'group_count': 4, 'left_context_mode': 'start_left_sign4', 'left_context_count': 4, 'num_position_group_context_codes': 12288, 'min_code_length': 1, 'max_code_length': 14, 'mean_code_length_unweighted': 6.5782216389973955}
- semantic_payload_bpp_mean: 0.008997101714645607
- detail_payload_bpp_mean: 0.005001510398975317
- actual_payload_bpp_mean: 0.013998612113620924
- paper_bpp_mean: 0.013998612113620924
- total_payload_bpp_mean: 0.013998612113620924
- debug_semantic_only_full_stream_bpp_mean: 0.023462433745895607
- debug_full_stream_bpp_mean: 0.029593094535495922
- semantic_only_full_stream_bpp_mean: 0.023462433745895607
- stage3_full_stream_bpp_mean: 0.029593094535495922
- semantic_payload_bytes_mean: 294.81702898550725
- detail_payload_bytes_mean: 163.8894927536232
- stage3_stream_bytes_mean: 969.7065217391304
- semantic_only_psnr_mean: 21.5603153049082
- semantic_only_l1_mean: 0.05815977243490625
- semantic_only_ms_ssim_mean: 0.7302724557483325
- stage3_psnr_mean: 21.879366406496022
- stage3_l1_mean: 0.05593222832860614
- stage3_ms_ssim_mean: 0.7342420197429432
- semantic_topk_hit_rate_mean: 0.9351152060688406
- semantic_token_roundtrip_mean: 1.0
- detail_code_roundtrip_mean: 1.0
- residual_grid_abs_mean_mean: 0.01964624380554491
- residual_grid_std_mean: 0.02472505450208226
- residual_grid_clip_ratio_mean: 1.1794233443282977e-05
- detail_code_entropy_bits_mean: 1.6657307444923166
- semantic_only_lpips_alex_mean: 0.5785921476646394
- semantic_only_dists_mean: 0.35848925295083417
- stage3_lpips_alex_mean: 0.5726936400085147
- stage3_dists_mean: 0.3555015931310861
- stage4_psnr_mean: 21.815504785897076
- stage4_l1_mean: 0.056288384857749486
- stage4_ms_ssim_mean: 0.7317225789192362
- stage4_refiner_residual_abs_mean_mean: 0.004099732473453817
- stage4_lpips_alex_mean: 0.49856347337608103
- stage4_dists_mean: 0.3370046500062597
- stage3_psnr_delta_vs_semantic_only: 0.3190511015878208
- stage3_l1_delta_vs_semantic_only: -0.0022275441063001056
- stage3_ms_ssim_delta_vs_semantic_only: 0.003969563994610681
- stage3_lpips_alex_delta_vs_semantic_only: -0.005898507656124652
- stage3_dists_delta_vs_semantic_only: -0.0029876598197480786
- stage4_psnr_delta_vs_stage3: -0.06386162059894573
- stage4_l1_delta_vs_stage3: 0.00035615652914334356
- stage4_ms_ssim_delta_vs_stage3: -0.002519440823707031
- stage4_lpips_alex_delta_vs_stage3: -0.07413016663243371
- stage4_dists_delta_vs_stage3: -0.018496943124826415
- all_semantic_tokens_roundtrip: True
- all_detail_codes_roundtrip: True
- roundtrip_failure_count: 0
- roundtrip_failures: []
- per_image_metrics: results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_ckpt250sweep_step500_cod512_full552_eval/per_image_metrics.jsonl
- reconstruction_manifest: results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_ckpt250sweep_step500_cod512_full552_eval/reconstructions/manifest.jsonl
- reconstruction_count: 552
- reconstruction_dirs: {'reference': 'results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_ckpt250sweep_step500_cod512_full552_eval/reconstructions/reference', 'semantic_only': 'results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_ckpt250sweep_step500_cod512_full552_eval/reconstructions/semantic_only', 'stage3': 'results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_ckpt250sweep_step500_cod512_full552_eval/reconstructions/stage3', 'stage4': 'results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_ckpt250sweep_step500_cod512_full552_eval/reconstructions/stage4'}

## Analysis

This is a strict actual-bitstream evaluation. The reported bpp is
`actual_payload_bpp / paper_bpp`: entropy-coded semantic and detail payloads
only. The fixed decoder-side Stage 4 refiner checkpoint is not image-specific
side information, so it does not change payload bpp.

Dataset-level image metrics:

| dataset | n | actual bpp | dPSNR | dMS-SSIM | dLPIPS | dDISTS | LPIPS wins | DISTS wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Kodak24 | 24 | 0.014120738 | -0.049467 | -0.002532 | -0.092378 | -0.021814 | 24/24 | 23/24 |
| CLIC2020 test | 428 | 0.013746172 | -0.065516 | -0.002552 | -0.071146 | -0.018188 | 423/428 | 385/428 |
| DIV2K val | 100 | 0.015049744 | -0.060235 | -0.002378 | -0.082522 | -0.019025 | 100/100 | 99/100 |

GenCodec public-script patch FID was computed separately after this evaluation
with `torchmetrics.image.fid.FrechetInceptionDistance`, Kodak64/non-Kodak256
patches, and split=2 half-shift. This is for FID only; PSNR, LPIPS, and DISTS
remain image-level.

| dataset | Stage3 FID | Stage4 FID | delta | patches |
|---|---:|---:|---:|---:|
| Kodak24 | 215.036469 | 177.938614 | -37.097855 | 2712 |
| CLIC2020 test | 147.601395 | 170.253677 | +22.652283 | 2140 |
| DIV2K val | 235.171097 | 262.204926 | +27.033829 | 500 |

Decision: do not promote this Stage 4 checkpoint. It improves LPIPS/DISTS
broadly at unchanged actual payload bpp and improves Kodak FID, but it fails
the full-dataset FID gate on CLIC2020 test and DIV2K val.

## Artifacts

- summary: `results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_ckpt250sweep_step500_cod512_full552_eval/summary.json`
- per_image_metrics: `results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_ckpt250sweep_step500_cod512_full552_eval/per_image_metrics.jsonl`
- sample: `results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_ckpt250sweep_step500_cod512_full552_eval/stage3_uniform_residual_grid.png`
- reconstruction_manifest: `results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_ckpt250sweep_step500_cod512_full552_eval/reconstructions/manifest.jsonl`
