# 20260627_stage2_position_huffman_backoff256_512calib_kodak_eval

Date: 2026-06-27T09:45:33

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage2_static_huffman_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --prior outputs/stage2_semantic_entropy/20260627_stage2_position_huffman_prior_512calib_backoff256_from_stage1_best/static_position_huffman_prior.json --image-root /dpl/kodak --max-images 24 --crop-size 256 --batch-size 4 --num-workers 4 --save-bitstreams --wandb-mode offline --run-name 20260627_stage2_position_huffman_backoff256_512calib_kodak_eval
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- prior: outputs/stage2_semantic_entropy/20260627_stage2_position_huffman_prior_512calib_backoff256_from_stage1_best/static_position_huffman_prior.json
- num_images: 24
- crop_size: 256
- codebook_size: 8192
- static_position_huffman_payload_bpp_mean: 0.013514200846354166
- static_position_huffman_actual_bpp_mean: 0.07711283365885417
- static_position_huffman_stream_bytes_mean: 631.7083333333334
- static_position_huffman_payload_bytes_mean: 110.70833333333333
- static_position_huffman_psnr_mean: 20.736252387364704
- static_position_huffman_l1_mean: 0.06508396441737811
- static_position_huffman_ms_ssim_mean: 0.6757684648036957
- static_position_huffman_roundtrip_mean: 1.0
- static_position_huffman_all_tokens_roundtrip: True
- fixed_bits_payload_bpp_mean: 0.0126953125
- fixed_bits_actual_bpp_mean: 0.07470703125
- fixed_bits_stream_bytes_mean: 612.0
- fixed_bits_payload_bytes_mean: 104.0
- fixed_bits_psnr_mean: 20.736252387364704
- fixed_bits_l1_mean: 0.06508396441737811
- fixed_bits_ms_ssim_mean: 0.6757684648036957
- fixed_bits_roundtrip_mean: 1.0
- fixed_bits_all_tokens_roundtrip: True
- static_position_huffman_payload_bpp_delta_vs_fixed_bits: 0.0008188883463541661

## Artifacts

- summary: `results/bitstreams/stage2_static_huffman/20260627_stage2_position_huffman_backoff256_512calib_kodak_eval/summary.json`
- output_dir: `results/bitstreams/stage2_static_huffman/20260627_stage2_position_huffman_backoff256_512calib_kodak_eval`
- streams: `results/bitstreams/stage2_static_huffman/20260627_stage2_position_huffman_backoff256_512calib_kodak_eval/streams`
