# 20260627_stage3_uniform_residual_d32_b4_r025_zlib_kodak_eval_stats

Date: 2026-06-27T11:37:15

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage3_uniform_residual_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --semantic-prior outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json --image-root /dpl/kodak --max-images 24 --batch-size 4 --num-workers 4 --detail-downsample-factor 32 --detail-bits 4 --detail-range 0.25 --detail-codec zlib_fixed_bits --wandb-mode offline --run-name 20260627_stage3_uniform_residual_d32_b4_r025_zlib_kodak_eval_stats
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- semantic_prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt
- num_images: 24
- crop_size: 256
- semantic_topk: 512
- detail_downsample_factor: 32
- detail_shape: [3, 8, 8]
- detail_bits: 4
- detail_range: 0.25
- detail_codec: zlib_fixed_bits
- residual_code: {'codec': 'uniform_residual_grid', 'version': 0, 'bits': 4, 'value_range': 0.25, 'payload_codec': 'zlib_fixed_bits'}
- semantic_payload_bpp_mean: 0.010721842447916666
- detail_payload_bpp_mean: 0.008758544921875
- total_payload_bpp_mean: 0.019480387369791668
- semantic_only_full_stream_bpp_mean: 0.07080586751302083
- stage3_full_stream_bpp_mean: 0.08090718587239583
- semantic_payload_bytes_mean: 87.83333333333333
- detail_payload_bytes_mean: 71.75
- stage3_stream_bytes_mean: 662.7916666666666
- semantic_only_psnr_mean: 20.736252387364704
- semantic_only_l1_mean: 0.06508396441737811
- semantic_only_ms_ssim_mean: 0.6757684648036957
- stage3_psnr_mean: 21.106779257456463
- stage3_l1_mean: 0.06182829748528699
- stage3_ms_ssim_mean: 0.6804229070742925
- semantic_topk_hit_rate_mean: 0.6217447916666666
- semantic_token_roundtrip_mean: 1.0
- detail_code_roundtrip_mean: 1.0
- residual_grid_abs_mean_mean: 0.020220475426564615
- residual_grid_std_mean: 0.02510768446760873
- residual_grid_clip_ratio_mean: 0.0
- detail_code_entropy_bits_mean: 1.6878134657939274
- stage3_psnr_delta_vs_semantic_only: 0.37052687009175855
- stage3_l1_delta_vs_semantic_only: -0.003255666932091124
- stage3_ms_ssim_delta_vs_semantic_only: 0.004654442270596859
- all_semantic_tokens_roundtrip: True
- all_detail_codes_roundtrip: True
- roundtrip_failure_count: 0
- roundtrip_failures: []

## Artifacts

- summary: `results/bitstreams/stage3_uniform_residual/20260627_stage3_uniform_residual_d32_b4_r025_zlib_kodak_eval_stats/summary.json`
- sample: `results/bitstreams/stage3_uniform_residual/20260627_stage3_uniform_residual_d32_b4_r025_zlib_kodak_eval_stats/stage3_uniform_residual_grid.png`
