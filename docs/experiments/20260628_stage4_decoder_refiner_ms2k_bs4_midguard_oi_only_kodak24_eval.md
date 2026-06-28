# 20260628_stage4_decoder_refiner_ms2k_bs4_midguard_oi_only_kodak24_eval

Date: 2026-06-28T13:10:31

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage3_uniform_residual_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt --semantic-prior outputs/stage2_learned_entropy/20260628_stage2_learned_topk2048_escape_huffman_fit_32768tokens_crop512_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json --token-prior-checkpoint checkpoints/stage2_token_prior/20260628_stage2_token_prior_crop512_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_amp_es.pt --detail-codec semantic_position_leftctx_huffman --detail-prior outputs/stage3_residual_entropy/20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_crop512_8192calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json --eval-protocol cod_reproduction_512 --eval-dataset kodak --crop-size 512 --batch-size 1 --detail-downsample-factor 32 --detail-bits 4 --detail-range 0.25 --detail-gain 0.5 --decoder-postprocess unsharp3x3 --decoder-postprocess-strength 0.15 --decoder-refiner-checkpoint checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms2k_bs4_anchor025_rs005_tv005_chroma010_oi_only.pt --compute-perceptual --save-reconstructions --save-reconstruction-limit 24 --save-reconstruction-triptychs --run-name 20260628_stage4_decoder_refiner_ms2k_bs4_midguard_oi_only_kodak24_eval --wandb-mode offline
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt
- semantic_prior: outputs/stage2_learned_entropy/20260628_stage2_learned_topk2048_escape_huffman_fit_32768tokens_crop512_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json
- detail_prior: outputs/stage3_residual_entropy/20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_crop512_8192calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260628_stage2_token_prior_crop512_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_amp_es.pt
- num_images: 24
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
- decoder_refiner_checkpoint: checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms2k_bs4_anchor025_rs005_tv005_chroma010_oi_only.pt
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
- eval_datasets: ['kodak']
- eval_image_roots: ['/dpl/kodak']
- eval_protocol_summary: {'name': 'cod_reproduction_512', 'display_name': 'CoD paper reproduction protocol, 512x512', 'dataset_keys': ['kodak'], 'dataset_counts': {'kodak': 24}, 'expected_counts': {'kodak': 24}, 'count_status': {'kodak': 'ok'}, 'total_images': 24, 'primary_metrics': ['PSNR', 'LPIPS', 'DISTS', 'FID'], 'bpp_metric': 'actual_payload_bpp for CoSER; official nominal/file bpp must be labeled for external codecs', 'default_crop_size': 512, 'notes': ['Primary CoSER low-bitrate generative comparison protocol for CoD-style tables.', 'Resize if needed and center-crop every image to 512x512 before evaluation.', 'Use CLIC2020 test 428 as professional/test plus mobile/test.', 'Use DIV2K validation images 0801-0900, not the first 100 files from a mixed DIV2K root.', 'For CoD paper patch FID, run per dataset and label exact settings: Kodak512 uses 64px patches with split=2; CLIC2020 uses 128px patches with split=2.', 'DIV2K patch FID should be reported only when the chosen patch size/backend is explicitly labeled.', 'Patch-based or overlapped-patch FID must label patch size, shift/split count, and backend.'], 'datasets': [{'key': 'kodak', 'display_name': 'Kodak 24', 'count': 24, 'expected_count': 24, 'status': 'ok', 'source_roots': ['/dpl/kodak'], 'first_path': '/dpl/kodak/kodim01.png', 'last_path': '/dpl/kodak/kodim24.png', 'notes': []}]}
- deterministic_eval: True
- compute_perceptual: True
- save_reconstructions: True
- save_reconstruction_limit: 24
- save_reconstruction_triptychs: True
- residual_code: {'codec': 'static_residual_grid_semantic_position_leftctx_huffman', 'version': 0, 'bits': 4, 'value_range': 0.25, 'payload_codec': 'semantic_position_leftctx_huffman', 'detail_shape': [3, 16, 16], 'semantic_shape': [16, 16], 'group_count': 4, 'left_context_mode': 'start_left_sign4', 'left_context_count': 4, 'num_position_group_context_codes': 12288, 'min_code_length': 1, 'max_code_length': 14, 'mean_code_length_unweighted': 6.5782216389973955}
- semantic_payload_bpp_mean: 0.009373982747395834
- detail_payload_bpp_mean: 0.004746754964192708
- actual_payload_bpp_mean: 0.014120737711588541
- paper_bpp_mean: 0.014120737711588541
- total_payload_bpp_mean: 0.014120737711588541
- debug_semantic_only_full_stream_bpp_mean: 0.023839314778645832
- debug_full_stream_bpp_mean: 0.029715220133463543
- semantic_only_full_stream_bpp_mean: 0.023839314778645832
- stage3_full_stream_bpp_mean: 0.029715220133463543
- semantic_payload_bytes_mean: 307.1666666666667
- detail_payload_bytes_mean: 155.54166666666666
- stage3_stream_bytes_mean: 973.7083333333334
- semantic_only_psnr_mean: 21.309127410252888
- semantic_only_l1_mean: 0.057320320357879005
- semantic_only_ms_ssim_mean: 0.7173154527942339
- stage3_psnr_mean: 21.562602996826172
- stage3_l1_mean: 0.0549161274296542
- stage3_ms_ssim_mean: 0.7215800558527311
- semantic_topk_hit_rate_mean: 0.9140625
- semantic_token_roundtrip_mean: 1.0
- detail_code_roundtrip_mean: 1.0
- residual_grid_abs_mean_mean: 0.018194327829405665
- residual_grid_std_mean: 0.02348115904411922
- residual_grid_clip_ratio_mean: 0.0
- detail_code_entropy_bits_mean: 1.6158534189065297
- semantic_only_lpips_alex_mean: 0.640996819982926
- semantic_only_dists_mean: 0.37833591798941296
- stage3_lpips_alex_mean: 0.6338581492503484
- stage3_dists_mean: 0.37536749243736267
- stage4_psnr_mean: 21.44454534848531
- stage4_l1_mean: 0.05589015440394481
- stage4_ms_ssim_mean: 0.7136281877756119
- stage4_refiner_residual_abs_mean_mean: 0.008098631767400851
- stage4_lpips_alex_mean: 0.48401108011603355
- stage4_dists_mean: 0.33342524866263074
- stage3_psnr_delta_vs_semantic_only: 0.25347558657328406
- stage3_l1_delta_vs_semantic_only: -0.002404192928224802
- stage3_ms_ssim_delta_vs_semantic_only: 0.004264603058497185
- stage3_lpips_alex_delta_vs_semantic_only: -0.007138670732577568
- stage3_dists_delta_vs_semantic_only: -0.002968425552050291
- stage4_psnr_delta_vs_stage3: -0.11805764834086219
- stage4_l1_delta_vs_stage3: 0.0009740269742906094
- stage4_ms_ssim_delta_vs_stage3: -0.007951868077119228
- stage4_lpips_alex_delta_vs_stage3: -0.1498470691343149
- stage4_dists_delta_vs_stage3: -0.041942243774731935
- all_semantic_tokens_roundtrip: True
- all_detail_codes_roundtrip: True
- roundtrip_failure_count: 0
- roundtrip_failures: []
- per_image_metrics: results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_ms2k_bs4_midguard_oi_only_kodak24_eval/per_image_metrics.jsonl
- reconstruction_manifest: results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_ms2k_bs4_midguard_oi_only_kodak24_eval/reconstructions/manifest.jsonl
- reconstruction_count: 24
- reconstruction_dirs: {'reference': 'results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_ms2k_bs4_midguard_oi_only_kodak24_eval/reconstructions/reference', 'semantic_only': 'results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_ms2k_bs4_midguard_oi_only_kodak24_eval/reconstructions/semantic_only', 'stage3': 'results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_ms2k_bs4_midguard_oi_only_kodak24_eval/reconstructions/stage3', 'stage4': 'results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_ms2k_bs4_midguard_oi_only_kodak24_eval/reconstructions/stage4', 'triptych': 'results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_ms2k_bs4_midguard_oi_only_kodak24_eval/reconstructions/triptych'}

## Analysis Notes

- paper_bpp / actual_payload_bpp: 0.014120738
- roundtrip_failure_count: 0
- Stage4 delta vs Stage3: PSNR -0.118058, MS-SSIM -0.007952, LPIPS -0.149847, DISTS -0.041942
- per-image wins: LPIPS 24/24, DISTS 24/24, PSNR 0/24, MS-SSIM 0/24
- CoD-style Kodak patch-FID, torchmetrics, patch64, split2: Stage3 215.036469, Stage4 165.095108
- Interpretation: strong Kodak perception/FID gain at unchanged payload bpp, but fidelity loss is much larger than the 300-step balanced checkpoint.

## Artifacts

- summary: `results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_ms2k_bs4_midguard_oi_only_kodak24_eval/summary.json`
- per_image_metrics: `results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_ms2k_bs4_midguard_oi_only_kodak24_eval/per_image_metrics.jsonl`
- sample: `results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_ms2k_bs4_midguard_oi_only_kodak24_eval/stage3_uniform_residual_grid.png`
- reconstruction_manifest: `results/bitstreams/stage3_uniform_residual/20260628_stage4_decoder_refiner_ms2k_bs4_midguard_oi_only_kodak24_eval/reconstructions/manifest.jsonl`
