# 20260627_stage1_semantic_vq_5k_hardvq_vq010_usage001_l1only_kodak_actual_bitstream_eval

Date: 2026-06-27T03:57:34

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage1_semantic_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --image-root /dpl/kodak --batch-size 4 --num-workers 4 --max-images 24 --crop-size 256 --save-bitstreams --wandb-mode offline --run-name 20260627_stage1_semantic_vq_5k_hardvq_vq010_usage001_l1only_kodak_actual_bitstream_eval
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- num_images: 24
- crop_size: 256
- semantic_tokens_per_image: 64
- codebook_size: 8192
- raw_uint16_be_actual_bpp_mean: 0.07763671875
- raw_uint16_be_token_payload_bpp_mean: 0.015625
- raw_uint16_be_container_overhead_bpp_mean: 0.06201171875
- raw_uint16_be_stream_bytes_mean: 636.0
- raw_uint16_be_token_payload_bytes_mean: 128.0
- raw_uint16_be_container_overhead_bytes_mean: 508.0
- raw_uint16_be_psnr_mean: 20.736252387364704
- raw_uint16_be_l1_mean: 0.06508396441737811
- raw_uint16_be_ms_ssim_mean: 0.6757684648036957
- raw_uint16_be_token_roundtrip_mean: 1.0
- raw_uint16_be_all_tokens_roundtrip: True
- fixed_bits_actual_bpp_mean: 0.0743408203125
- fixed_bits_token_payload_bpp_mean: 0.0126953125
- fixed_bits_container_overhead_bpp_mean: 0.0616455078125
- fixed_bits_stream_bytes_mean: 609.0
- fixed_bits_token_payload_bytes_mean: 104.0
- fixed_bits_container_overhead_bytes_mean: 505.0
- fixed_bits_psnr_mean: 20.736252387364704
- fixed_bits_l1_mean: 0.06508396441737811
- fixed_bits_ms_ssim_mean: 0.6757684648036957
- fixed_bits_token_roundtrip_mean: 1.0
- fixed_bits_all_tokens_roundtrip: True
- zlib_uint16_be_actual_bpp_mean: 0.078887939453125
- zlib_uint16_be_token_payload_bpp_mean: 0.016754150390625
- zlib_uint16_be_container_overhead_bpp_mean: 0.0621337890625
- zlib_uint16_be_stream_bytes_mean: 646.25
- zlib_uint16_be_token_payload_bytes_mean: 137.25
- zlib_uint16_be_container_overhead_bytes_mean: 509.0
- zlib_uint16_be_psnr_mean: 20.736252387364704
- zlib_uint16_be_l1_mean: 0.06508396441737811
- zlib_uint16_be_ms_ssim_mean: 0.6757684648036957
- zlib_uint16_be_token_roundtrip_mean: 1.0
- zlib_uint16_be_all_tokens_roundtrip: True
- zlib_fixed_bits_actual_bpp_mean: 0.0762786865234375
- zlib_fixed_bits_token_payload_bpp_mean: 0.0140228271484375
- zlib_fixed_bits_container_overhead_bpp_mean: 0.062255859375
- zlib_fixed_bits_stream_bytes_mean: 624.875
- zlib_fixed_bits_token_payload_bytes_mean: 114.875
- zlib_fixed_bits_container_overhead_bytes_mean: 510.0
- zlib_fixed_bits_psnr_mean: 20.736252387364704
- zlib_fixed_bits_l1_mean: 0.06508396441737811
- zlib_fixed_bits_ms_ssim_mean: 0.6757684648036957
- zlib_fixed_bits_token_roundtrip_mean: 1.0
- zlib_fixed_bits_all_tokens_roundtrip: True

## Artifacts

- summary: `results/bitstreams/stage1_semantic_vq/20260627_stage1_semantic_vq_5k_hardvq_vq010_usage001_l1only_kodak_actual_bitstream_eval/summary.json`
- output_dir: `results/bitstreams/stage1_semantic_vq/20260627_stage1_semantic_vq_5k_hardvq_vq010_usage001_l1only_kodak_actual_bitstream_eval`
- streams: `results/bitstreams/stage1_semantic_vq/20260627_stage1_semantic_vq_5k_hardvq_vq010_usage001_l1only_kodak_actual_bitstream_eval/streams`
