# 20260627_stage3_residual_huffman_d32_b4_r050_4096calib_kodak_eval

Date: 2026-06-27T11:53:48

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage3_uniform_residual_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --semantic-prior outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json --detail-codec huffman --detail-prior outputs/stage3_residual_entropy/20260627_stage3_residual_huffman_d32_b4_r050_4096calib_oi_div2k/static_residual_grid_huffman_prior.json --image-root /dpl/kodak --max-images 24 --batch-size 4 --num-workers 4 --detail-downsample-factor 32 --detail-bits 4 --detail-range 0.5 --wandb-mode offline --run-name 20260627_stage3_residual_huffman_d32_b4_r050_4096calib_kodak_eval
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- semantic_prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json
- detail_prior: outputs/stage3_residual_entropy/20260627_stage3_residual_huffman_d32_b4_r050_4096calib_oi_div2k/static_residual_grid_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt
- num_images: 24
- crop_size: 256
- semantic_topk: 512
- detail_downsample_factor: 32
- detail_shape: [3, 8, 8]
- detail_bits: 4
- detail_range: 0.5
- detail_codec: huffman
- residual_code: {'codec': 'static_residual_grid_huffman', 'version': 0, 'bits': 4, 'value_range': 0.5, 'payload_codec': 'huffman', 'huffman': {'codec': 'static_huffman', 'version': 0, 'codebook_size': 16, 'code_lengths': [14, 11, 10, 10, 8, 6, 4, 2, 1, 3, 5, 7, 10, 12, 13, 14]}}
- semantic_payload_bpp_mean: 0.010721842447916666
- detail_payload_bpp_mean: 0.004425048828125
- total_payload_bpp_mean: 0.015146891276041666
- semantic_only_full_stream_bpp_mean: 0.07080586751302083
- stage3_full_stream_bpp_mean: 0.07657368977864583
- semantic_payload_bytes_mean: 87.83333333333333
- detail_payload_bytes_mean: 36.25
- stage3_stream_bytes_mean: 627.2916666666666
- semantic_only_psnr_mean: 20.736252387364704
- semantic_only_l1_mean: 0.06508396441737811
- semantic_only_ms_ssim_mean: 0.6757684648036957
- stage3_psnr_mean: 20.967170238494873
- stage3_l1_mean: 0.06365959501514833
- stage3_ms_ssim_mean: 0.6753987868626913
- semantic_topk_hit_rate_mean: 0.6217447916666666
- semantic_token_roundtrip_mean: 1.0
- detail_code_roundtrip_mean: 1.0
- residual_grid_abs_mean_mean: 0.020220475426564615
- residual_grid_std_mean: 0.02510768446760873
- residual_grid_clip_ratio_mean: 0.0
- detail_code_entropy_bits_mean: 1.109691080947717
- stage3_psnr_delta_vs_semantic_only: 0.23091785113016883
- stage3_l1_delta_vs_semantic_only: -0.001424369402229786
- stage3_ms_ssim_delta_vs_semantic_only: -0.0003696779410043982
- all_semantic_tokens_roundtrip: True
- all_detail_codes_roundtrip: True
- roundtrip_failure_count: 0
- roundtrip_failures: []

## Artifacts

- summary: `results/bitstreams/stage3_uniform_residual/20260627_stage3_residual_huffman_d32_b4_r050_4096calib_kodak_eval/summary.json`
- sample: `results/bitstreams/stage3_uniform_residual/20260627_stage3_residual_huffman_d32_b4_r050_4096calib_kodak_eval/stage3_uniform_residual_grid.png`
