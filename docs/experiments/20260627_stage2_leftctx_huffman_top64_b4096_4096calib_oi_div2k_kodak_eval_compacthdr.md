# 20260627_stage2_leftctx_huffman_top64_b4096_4096calib_oi_div2k_kodak_eval_compacthdr

Date: 2026-06-27T10:02:32

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage2_static_huffman_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --prior outputs/stage2_semantic_entropy/20260627_stage2_leftctx_huffman_top64_b4096_4096calib_oi_div2k_from_stage1_best/static_left_context_huffman_prior.json --image-root /dpl/kodak --max-images 24 --batch-size 4 --num-workers 4 --wandb-mode offline --run-name 20260627_stage2_leftctx_huffman_top64_b4096_4096calib_oi_div2k_kodak_eval_compacthdr
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- prior: outputs/stage2_semantic_entropy/20260627_stage2_leftctx_huffman_top64_b4096_4096calib_oi_div2k_from_stage1_best/static_left_context_huffman_prior.json
- num_images: 24
- crop_size: 256
- codebook_size: 8192
- static_left_context_huffman_payload_bpp_mean: 0.011728922526041666
- static_left_context_huffman_actual_bpp_mean: 0.07143656412760417
- static_left_context_huffman_stream_bytes_mean: 585.2083333333334
- static_left_context_huffman_payload_bytes_mean: 96.08333333333333
- static_left_context_huffman_psnr_mean: 20.736252387364704
- static_left_context_huffman_l1_mean: 0.06508396441737811
- static_left_context_huffman_ms_ssim_mean: 0.6757684648036957
- static_left_context_huffman_roundtrip_mean: 1.0
- static_left_context_huffman_all_tokens_roundtrip: True
- fixed_bits_payload_bpp_mean: 0.0126953125
- fixed_bits_actual_bpp_mean: 0.072509765625
- fixed_bits_stream_bytes_mean: 594.0
- fixed_bits_payload_bytes_mean: 104.0
- fixed_bits_psnr_mean: 20.736252387364704
- fixed_bits_l1_mean: 0.06508396441737811
- fixed_bits_ms_ssim_mean: 0.6757684648036957
- fixed_bits_roundtrip_mean: 1.0
- fixed_bits_all_tokens_roundtrip: True
- static_left_context_huffman_payload_bpp_delta_vs_fixed_bits: -0.0009663899739583339

## Artifacts

- summary: `results/bitstreams/stage2_static_huffman/20260627_stage2_leftctx_huffman_top64_b4096_4096calib_oi_div2k_kodak_eval_compacthdr/summary.json`
- output_dir: `results/bitstreams/stage2_static_huffman/20260627_stage2_leftctx_huffman_top64_b4096_4096calib_oi_div2k_kodak_eval_compacthdr`
- streams: ``
