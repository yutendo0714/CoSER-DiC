# 20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_clicval64_eval_compacthdr

Date: 2026-06-27T12:41:42

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage3_uniform_residual_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --semantic-prior outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json --image-root /dpl/clic/professional/valid --max-images 64 --batch-size 4 --num-workers 4 --detail-downsample-factor 32 --detail-bits 5 --detail-range 0.25 --detail-codec position_huffman --detail-prior outputs/stage3_residual_entropy/20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_oi_div2k/static_residual_grid_position_huffman_prior.json --stream-header-codec compact --wandb-mode offline --run-name 20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_clicval64_eval_compacthdr
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- semantic_prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json
- detail_prior: outputs/stage3_residual_entropy/20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_oi_div2k/static_residual_grid_position_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt
- num_images: 41
- crop_size: 256
- semantic_topk: 512
- detail_downsample_factor: 32
- detail_shape: [3, 8, 8]
- detail_bits: 5
- detail_range: 0.25
- detail_codec: position_huffman
- stream_header_codec: compact
- residual_code: {'codec': 'static_residual_grid_position_huffman', 'version': 0, 'bits': 5, 'value_range': 0.25, 'payload_codec': 'position_huffman', 'detail_shape': [3, 8, 8], 'num_position_codes': 192, 'min_code_length': 2, 'max_code_length': 13, 'mean_code_length_unweighted': 8.402180989583334}
- semantic_payload_bpp_mean: 0.009905559260670731
- detail_payload_bpp_mean: 0.007967320884146341
- total_payload_bpp_mean: 0.017872880144817072
- semantic_only_full_stream_bpp_mean: 0.026995403010670733
- stage3_full_stream_bpp_mean: 0.036183427019817076
- semantic_payload_bytes_mean: 81.14634146341463
- detail_payload_bytes_mean: 65.26829268292683
- stage3_stream_bytes_mean: 296.4146341463415
- semantic_only_psnr_mean: 22.46947255948695
- semantic_only_l1_mean: 0.05876295443442536
- semantic_only_ms_ssim_mean: 0.7211727790716218
- stage3_psnr_mean: 23.186546255902545
- stage3_l1_mean: 0.056094700834009706
- stage3_ms_ssim_mean: 0.724869132768817
- semantic_topk_hit_rate_mean: 0.7164634146341463
- semantic_token_roundtrip_mean: 1.0
- detail_code_roundtrip_mean: 1.0
- residual_grid_abs_mean_mean: 0.018476074039027457
- residual_grid_std_mean: 0.022409117280891757
- residual_grid_clip_ratio_mean: 0.0
- detail_code_entropy_bits_mean: 2.3939700882609296
- stage3_psnr_delta_vs_semantic_only: 0.7170736964155964
- stage3_l1_delta_vs_semantic_only: -0.002668253600415657
- stage3_ms_ssim_delta_vs_semantic_only: 0.0036963536971952315
- all_semantic_tokens_roundtrip: True
- all_detail_codes_roundtrip: True
- roundtrip_failure_count: 0
- roundtrip_failures: []

## Artifacts

- summary: `results/bitstreams/stage3_uniform_residual/20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_clicval64_eval_compacthdr/summary.json`
- sample: `results/bitstreams/stage3_uniform_residual/20260627_stage3_residual_poshuff_d32_b5_r025_4096calib_clicval64_eval_compacthdr/stage3_uniform_residual_grid.png`
