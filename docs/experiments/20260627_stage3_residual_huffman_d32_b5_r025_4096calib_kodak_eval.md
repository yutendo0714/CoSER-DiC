# 20260627_stage3_residual_huffman_d32_b5_r025_4096calib_kodak_eval

Date: 2026-06-27T11:51:38

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage3_uniform_residual_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --semantic-prior outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json --detail-codec huffman --detail-prior outputs/stage3_residual_entropy/20260627_stage3_residual_huffman_d32_b5_r025_4096calib_oi_div2k/static_residual_grid_huffman_prior.json --image-root /dpl/kodak --max-images 24 --batch-size 4 --num-workers 4 --detail-downsample-factor 32 --detail-bits 5 --detail-range 0.25 --wandb-mode offline --run-name 20260627_stage3_residual_huffman_d32_b5_r025_4096calib_kodak_eval
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- semantic_prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json
- detail_prior: outputs/stage3_residual_entropy/20260627_stage3_residual_huffman_d32_b5_r025_4096calib_oi_div2k/static_residual_grid_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt
- num_images: 24
- crop_size: 256
- semantic_topk: 512
- detail_downsample_factor: 32
- detail_shape: [3, 8, 8]
- detail_bits: 5
- detail_range: 0.25
- detail_codec: huffman
- residual_code: {'codec': 'static_residual_grid_huffman', 'version': 0, 'bits': 5, 'value_range': 0.25, 'payload_codec': 'huffman', 'huffman': {'codec': 'static_huffman', 'version': 0, 'codebook_size': 32, 'code_lengths': [14, 15, 15, 14, 13, 12, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 14, 13]}}
- semantic_payload_bpp_mean: 0.010721842447916666
- detail_payload_bpp_mean: 0.0082855224609375
- total_payload_bpp_mean: 0.019007364908854168
- semantic_only_full_stream_bpp_mean: 0.07080586751302083
- stage3_full_stream_bpp_mean: 0.08043416341145833
- semantic_payload_bytes_mean: 87.83333333333333
- detail_payload_bytes_mean: 67.875
- stage3_stream_bytes_mean: 658.9166666666666
- semantic_only_psnr_mean: 20.736252387364704
- semantic_only_l1_mean: 0.06508396441737811
- semantic_only_ms_ssim_mean: 0.6757684648036957
- stage3_psnr_mean: 21.131991545359295
- stage3_l1_mean: 0.06152551124493281
- stage3_ms_ssim_mean: 0.6815250987807909
- semantic_topk_hit_rate_mean: 0.6217447916666666
- semantic_token_roundtrip_mean: 1.0
- detail_code_roundtrip_mean: 1.0
- residual_grid_abs_mean_mean: 0.020220475426564615
- residual_grid_std_mean: 0.02510768446760873
- residual_grid_clip_ratio_mean: 0.0
- detail_code_entropy_bits_mean: 2.5876581023136773
- stage3_psnr_delta_vs_semantic_only: 0.3957391579945906
- stage3_l1_delta_vs_semantic_only: -0.003558453172445304
- stage3_ms_ssim_delta_vs_semantic_only: 0.005756633977095249
- all_semantic_tokens_roundtrip: True
- all_detail_codes_roundtrip: True
- roundtrip_failure_count: 0
- roundtrip_failures: []

## Artifacts

- summary: `results/bitstreams/stage3_uniform_residual/20260627_stage3_residual_huffman_d32_b5_r025_4096calib_kodak_eval/summary.json`
- sample: `results/bitstreams/stage3_uniform_residual/20260627_stage3_residual_huffman_d32_b5_r025_4096calib_kodak_eval/stage3_uniform_residual_grid.png`
