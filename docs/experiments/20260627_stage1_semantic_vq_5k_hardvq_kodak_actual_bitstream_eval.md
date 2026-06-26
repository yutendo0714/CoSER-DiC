# 20260627_stage1_semantic_vq_5k_hardvq_kodak_actual_bitstream_eval

Date: 2026-06-27T03:25:23

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage1_semantic_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_l1.pt --image-root /dpl/kodak --batch-size 4 --num-workers 4 --max-images 24 --crop-size 256 --save-bitstreams --wandb-mode offline --run-name 20260627_stage1_semantic_vq_5k_hardvq_kodak_actual_bitstream_eval
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_l1.pt
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
- raw_uint16_be_psnr_mean: 20.122809648513794
- raw_uint16_be_l1_mean: 0.07248700394605596
- raw_uint16_be_ms_ssim_mean: 0.6373234825829664
- raw_uint16_be_token_roundtrip_mean: 1.0
- raw_uint16_be_all_tokens_roundtrip: True
- fixed_bits_actual_bpp_mean: 0.0743408203125
- fixed_bits_token_payload_bpp_mean: 0.0126953125
- fixed_bits_container_overhead_bpp_mean: 0.0616455078125
- fixed_bits_stream_bytes_mean: 609.0
- fixed_bits_token_payload_bytes_mean: 104.0
- fixed_bits_container_overhead_bytes_mean: 505.0
- fixed_bits_psnr_mean: 20.122809648513794
- fixed_bits_l1_mean: 0.07248700394605596
- fixed_bits_ms_ssim_mean: 0.6373234825829664
- fixed_bits_token_roundtrip_mean: 1.0
- fixed_bits_all_tokens_roundtrip: True
- zlib_uint16_be_actual_bpp_mean: 0.0787353515625
- zlib_uint16_be_token_payload_bpp_mean: 0.0166015625
- zlib_uint16_be_container_overhead_bpp_mean: 0.0621337890625
- zlib_uint16_be_stream_bytes_mean: 645.0
- zlib_uint16_be_token_payload_bytes_mean: 136.0
- zlib_uint16_be_container_overhead_bytes_mean: 509.0
- zlib_uint16_be_psnr_mean: 20.122809648513794
- zlib_uint16_be_l1_mean: 0.07248700394605596
- zlib_uint16_be_ms_ssim_mean: 0.6373234825829664
- zlib_uint16_be_token_roundtrip_mean: 1.0
- zlib_uint16_be_all_tokens_roundtrip: True
- zlib_fixed_bits_actual_bpp_mean: 0.0762939453125
- zlib_fixed_bits_token_payload_bpp_mean: 0.0140380859375
- zlib_fixed_bits_container_overhead_bpp_mean: 0.062255859375
- zlib_fixed_bits_stream_bytes_mean: 625.0
- zlib_fixed_bits_token_payload_bytes_mean: 115.0
- zlib_fixed_bits_container_overhead_bytes_mean: 510.0
- zlib_fixed_bits_psnr_mean: 20.122809648513794
- zlib_fixed_bits_l1_mean: 0.07248700394605596
- zlib_fixed_bits_ms_ssim_mean: 0.6373234825829664
- zlib_fixed_bits_token_roundtrip_mean: 1.0
- zlib_fixed_bits_all_tokens_roundtrip: True

## Artifacts

- summary: `results/bitstreams/stage1_semantic_vq/20260627_stage1_semantic_vq_5k_hardvq_kodak_actual_bitstream_eval/summary.json`
- output_dir: `results/bitstreams/stage1_semantic_vq/20260627_stage1_semantic_vq_5k_hardvq_kodak_actual_bitstream_eval`
- streams: `results/bitstreams/stage1_semantic_vq/20260627_stage1_semantic_vq_5k_hardvq_kodak_actual_bitstream_eval/streams`
