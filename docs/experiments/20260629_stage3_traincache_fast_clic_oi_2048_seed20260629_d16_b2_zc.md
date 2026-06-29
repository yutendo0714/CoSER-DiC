# 20260629_stage3_traincache_fast_clic_oi_2048_seed20260629_d16_b2_zc

Date: 2026-06-30T00:09:20

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/build_stage3_training_cache_fast.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt --image-root /dpl/clic/professional/train --image-root /dpl/clic/mobile/train --image-root /dpl/open-images-v6/train/data --crop-size 512 --max-images 2048 --shuffle-images --shuffle-seed 20260629 --batch-size 4 --num-workers 4 --detail-downsample-factor 16 --detail-bits 2 --detail-range 0.25 --detail-quantizer zero_centered --detail-codec semantic_position_leftctx_huffman --detail-prior outputs/stage3_residual_entropy/20260629_stage3_residual_semposleft_g4_sm0_d16_b2_zc_r025_crop512_2048calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json --run-name 20260629_stage3_traincache_fast_clic_oi_2048_seed20260629_d16_b2_zc --wandb-mode offline
```

## Summary

- semantic_only_psnr_mean: 21.498402379453182
- semantic_only_l1_mean: 0.05721563935458107
- semantic_only_ms_ssim_mean: 0.7390775525564095
- stage3_psnr_mean: 21.910340222530067
- stage3_l1_mean: 0.05518395462559056
- stage3_ms_ssim_mean: 0.7487120314472122
- residual_grid_abs_mean_mean: 0.028139100499174674
- residual_grid_std_mean: 0.03911742984018929
- residual_grid_clip_ratio_mean: 0.0006791750776500294
- cache_mode: fast_decoded_equivalent_no_entropy_roundtrip
- count: 2048
- crop_size: 512
- detail_downsample_factor: 16
- detail_bits: 2
- detail_range: 0.25
- detail_quantizer: zero_centered
- detail_gain: 1.0
- actual_payload_bpp: 
- paper_bpp: 
- bpp_note: not computed; this cache is for Stage4 training only. Use eval_stage3_uniform_residual_bitstream.py for actual payload bpp.

## Artifacts

- summary: `results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_2048_seed20260629_d16_b2_zc/summary.json`
- per_image_metrics: `results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_2048_seed20260629_d16_b2_zc/per_image_metrics.jsonl`
- reconstruction_manifest: `results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_2048_seed20260629_d16_b2_zc/reconstructions/manifest.jsonl`
- decoder_feature_cache: `results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_2048_seed20260629_d16_b2_zc/decoder_feature_cache`
- sample: `results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_2048_seed20260629_d16_b2_zc/stage3_training_cache_fast_grid.png`
