# 20260627_stage3_residual_hybrid_pos_semposg8_sm0_d32_b4_r025_8192calib_clicpro41_eval_compactv3_crc32

Date: 2026-06-27T14:42:52

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage3_uniform_residual_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --semantic-prior outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json --token-prior-checkpoint checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt --image-root /dpl/clic/professional/valid --crop-size 256 --max-images 64 --batch-size 4 --num-workers 4 --detail-downsample-factor 32 --detail-bits 4 --detail-range 0.25 --detail-codec hybrid_huffman --detail-prior outputs/stage3_residual_entropy/20260627_stage3_residual_hybrid_pos_semposg8_sm0_d32_b4_r025_8192calib_openimages_div2k/static_residual_grid_hybrid_huffman_prior.json --stream-header-codec compact --stream-checksum-codec crc32 --wandb-mode offline --run-name 20260627_stage3_residual_hybrid_pos_semposg8_sm0_d32_b4_r025_8192calib_clicpro41_eval_compactv3_crc32
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- semantic_prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json
- detail_prior: outputs/stage3_residual_entropy/20260627_stage3_residual_hybrid_pos_semposg8_sm0_d32_b4_r025_8192calib_openimages_div2k/static_residual_grid_hybrid_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt
- num_images: 41
- crop_size: 256
- semantic_topk: 512
- detail_downsample_factor: 32
- detail_shape: [3, 8, 8]
- detail_bits: 4
- detail_range: 0.25
- detail_codec: hybrid_huffman
- stream_header_codec: compact
- stream_checksum_codec: crc32
- residual_code: {'codec': 'static_residual_grid_hybrid_huffman', 'version': 0, 'bits': 4, 'value_range': 0.25, 'payload_codec': 'hybrid_huffman', 'position_code': {'codec': 'static_residual_grid_position_huffman', 'version': 0, 'bits': 4, 'value_range': 0.25, 'payload_codec': 'position_huffman', 'detail_shape': [3, 8, 8], 'num_position_codes': 192, 'min_code_length': 1, 'max_code_length': 14, 'mean_code_length_unweighted': 7.81640625}, 'semantic_position_code': {'codec': 'static_residual_grid_semantic_position_huffman', 'version': 0, 'bits': 4, 'value_range': 0.25, 'payload_codec': 'semantic_position_huffman', 'detail_shape': [3, 8, 8], 'semantic_shape': [8, 8], 'group_count': 8, 'num_position_group_codes': 1536, 'min_code_length': 1, 'max_code_length': 14, 'mean_code_length_unweighted': 7.513468424479167}}
- semantic_payload_bpp_mean: 0.009905559260670731
- detail_payload_bpp_mean: 0.005079315929878049
- total_payload_bpp_mean: 0.01498487519054878
- semantic_only_full_stream_bpp_mean: 0.014666301448170731
- stage3_full_stream_bpp_mean: 0.01998975800304878
- semantic_payload_bytes_mean: 81.14634146341463
- detail_payload_bytes_mean: 41.609756097560975
- stage3_stream_bytes_mean: 163.7560975609756
- semantic_only_psnr_mean: 22.46947255948695
- semantic_only_l1_mean: 0.05876295443442536
- semantic_only_ms_ssim_mean: 0.7211727790716218
- stage3_psnr_mean: 23.034366049417635
- stage3_l1_mean: 0.056612195810530244
- stage3_ms_ssim_mean: 0.7236354663604643
- semantic_topk_hit_rate_mean: 0.7164634146341463
- semantic_token_roundtrip_mean: 1.0
- detail_code_roundtrip_mean: 1.0
- residual_grid_abs_mean_mean: 0.018476074039027457
- residual_grid_std_mean: 0.022409117280891757
- residual_grid_clip_ratio_mean: 0.0
- detail_code_entropy_bits_mean: 1.5283508518847024
- detail_hybrid_semantic_mode_mean: 0.6341463414634146
- stage3_psnr_delta_vs_semantic_only: 0.5648934899306859
- stage3_l1_delta_vs_semantic_only: -0.0021507586238951187
- stage3_ms_ssim_delta_vs_semantic_only: 0.002462687288842469
- all_semantic_tokens_roundtrip: True
- all_detail_codes_roundtrip: True
- roundtrip_failure_count: 0
- roundtrip_failures: []

## Artifacts

- summary: `results/bitstreams/stage3_uniform_residual/20260627_stage3_residual_hybrid_pos_semposg8_sm0_d32_b4_r025_8192calib_clicpro41_eval_compactv3_crc32/summary.json`
- sample: `results/bitstreams/stage3_uniform_residual/20260627_stage3_residual_hybrid_pos_semposg8_sm0_d32_b4_r025_8192calib_clicpro41_eval_compactv3_crc32/stage3_uniform_residual_grid.png`
