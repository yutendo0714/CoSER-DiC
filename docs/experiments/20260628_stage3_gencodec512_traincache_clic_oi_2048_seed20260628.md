# 20260628_stage3_gencodec512_traincache_clic_oi_2048_seed20260628

Date: 2026-06-28T20:39:31

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage3_uniform_residual_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt --semantic-prior outputs/stage2_learned_entropy/20260628_stage2_learned_topk2048_escape_huffman_fit_32768tokens_crop512_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json --token-prior-checkpoint checkpoints/stage2_token_prior/20260628_stage2_token_prior_crop512_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_amp_es.pt --detail-prior outputs/stage3_residual_entropy/20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_crop512_8192calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json --image-root /dpl/clic/professional/train --image-root /dpl/clic/mobile/train --image-root /dpl/open-images-v6/train/data --crop-size 512 --max-images 2048 --shuffle-images --shuffle-seed 20260628 --batch-size 4 --num-workers 4 --detail-downsample-factor 32 --detail-bits 4 --detail-range 0.25 --detail-gain 1.0 --detail-codec semantic_position_leftctx_huffman --stream-header-codec json --stream-checksum-codec sha256 --save-reconstructions --save-decoder-feature-cache --output-dir results/bitstreams/stage3_uniform_residual --run-name 20260628_stage3_gencodec512_traincache_clic_oi_2048_seed20260628 --wandb-mode offline
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt
- semantic_prior: outputs/stage2_learned_entropy/20260628_stage2_learned_topk2048_escape_huffman_fit_32768tokens_crop512_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json
- detail_prior: outputs/stage3_residual_entropy/20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_crop512_8192calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260628_stage2_token_prior_crop512_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_amp_es.pt
- num_images: 2048
- crop_size: 512
- protocol_default_crop_size: None
- crop_size_matches_protocol_default: None
- semantic_topk: 2048
- semantic_topk_schedule: prefix_replay_decoder_safe
- detail_downsample_factor: 32
- detail_shape: [3, 16, 16]
- detail_bits: 4
- detail_range: 0.25
- detail_gain: 1.0
- decoder_postprocess: none
- decoder_postprocess_strength: 0.0
- decoder_refiner_checkpoint: 
- decoder_refiner_enabled: False
- decoder_refiner_config: {}
- decoder_refiner_payload_policy: 
- detail_codec: semantic_position_leftctx_huffman
- stream_header_codec: json
- stream_checksum_codec: sha256
- main_bpp_metric: actual_payload_bpp_mean
- paper_bpp_metric: paper_bpp_mean
- debug_bpp_metric: debug_full_stream_bpp_mean
- eval_protocol: manual_roots
- eval_datasets: []
- eval_image_roots: ['/dpl/clic/professional/train', '/dpl/clic/mobile/train', '/dpl/open-images-v6/train/data']
- eval_protocol_summary: {}
- deterministic_eval: True
- shuffle_images: True
- shuffle_seed: 20260628
- compute_perceptual: False
- save_reconstructions: True
- save_reconstruction_limit: 0
- save_reconstruction_triptychs: False
- save_decoder_feature_cache: True
- decoder_feature_cache_dir: results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_traincache_clic_oi_2048_seed20260628/decoder_feature_cache
- residual_code: {'codec': 'static_residual_grid_semantic_position_leftctx_huffman', 'version': 0, 'bits': 4, 'value_range': 0.25, 'payload_codec': 'semantic_position_leftctx_huffman', 'detail_shape': [3, 16, 16], 'semantic_shape': [16, 16], 'group_count': 4, 'left_context_mode': 'start_left_sign4', 'left_context_count': 4, 'num_position_group_context_codes': 12288, 'min_code_length': 1, 'max_code_length': 14, 'mean_code_length_unweighted': 6.5782216389973955}
- semantic_payload_bpp_mean: 0.009187504649162292
- detail_payload_bpp_mean: 0.005089268088340759
- actual_payload_bpp_mean: 0.014276772737503052
- paper_bpp_mean: 0.014276772737503052
- total_payload_bpp_mean: 0.014276772737503052
- debug_semantic_only_full_stream_bpp_mean: 0.023652836680412292
- debug_full_stream_bpp_mean: 0.029169350862503052
- semantic_only_full_stream_bpp_mean: 0.023652836680412292
- stage3_full_stream_bpp_mean: 0.029169350862503052
- semantic_payload_bytes_mean: 301.05615234375
- detail_payload_bytes_mean: 166.76513671875
- stage3_stream_bytes_mean: 955.8212890625
- semantic_only_psnr_mean: 21.58567892992869
- semantic_only_l1_mean: 0.05670959838425915
- semantic_only_ms_ssim_mean: 0.7391734145930968
- stage3_psnr_mean: 22.06874723592773
- stage3_l1_mean: 0.05368214301779517
- stage3_ms_ssim_mean: 0.7443766783253523
- semantic_topk_hit_rate_mean: 0.9252357482910156
- semantic_token_roundtrip_mean: 1.0
- detail_code_roundtrip_mean: 1.0
- residual_grid_abs_mean_mean: 0.019970675215517986
- residual_grid_std_mean: 0.026022305759397568
- residual_grid_clip_ratio_mean: 2.797444744828681e-05
- detail_code_entropy_bits_mean: 1.7198456322366837
- stage3_psnr_delta_vs_semantic_only: 0.4830683059990406
- stage3_l1_delta_vs_semantic_only: -0.0030274553664639825
- stage3_ms_ssim_delta_vs_semantic_only: 0.0052032637322554365
- all_semantic_tokens_roundtrip: True
- all_detail_codes_roundtrip: True
- roundtrip_failure_count: 0
- roundtrip_failures: []
- per_image_metrics: results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_traincache_clic_oi_2048_seed20260628/per_image_metrics.jsonl
- reconstruction_manifest: results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_traincache_clic_oi_2048_seed20260628/reconstructions/manifest.jsonl
- reconstruction_count: 2048
- reconstruction_dirs: {'reference': 'results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_traincache_clic_oi_2048_seed20260628/reconstructions/reference', 'semantic_only': 'results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_traincache_clic_oi_2048_seed20260628/reconstructions/semantic_only', 'stage3': 'results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_traincache_clic_oi_2048_seed20260628/reconstructions/stage3'}

## Artifacts

- summary: `results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_traincache_clic_oi_2048_seed20260628/summary.json`
- per_image_metrics: `results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_traincache_clic_oi_2048_seed20260628/per_image_metrics.jsonl`
- sample: `results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_traincache_clic_oi_2048_seed20260628/stage3_uniform_residual_grid.png`
- reconstruction_manifest: `results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_traincache_clic_oi_2048_seed20260628/reconstructions/manifest.jsonl`
