# 20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4

Date: 2026-06-29T01:22:29

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/build_stage3_training_cache_fast.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt --detail-prior outputs/stage3_residual_entropy/20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_crop512_8192calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json --image-root /dpl/clic/professional/train --image-root /dpl/clic/mobile/train --image-root /dpl/open-images-v6/train/data --crop-size 512 --max-images 8192 --shuffle-images --shuffle-seed 20260629 --batch-size 4 --num-workers 4 --detail-downsample-factor 32 --detail-bits 4 --detail-range 0.25 --detail-gain 1.0 --detail-codec semantic_position_leftctx_huffman --run-name 20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4 --wandb-mode offline
```

## Summary

- semantic_only_psnr_mean: 21.506887775380164
- semantic_only_l1_mean: 0.05716115177176562
- semantic_only_ms_ssim_mean: 0.7392380290475558
- stage3_psnr_mean: 21.98009490314871
- stage3_l1_mean: 0.05412892867758501
- stage3_ms_ssim_mean: 0.744562540017796
- residual_grid_abs_mean_mean: 0.020181254541796534
- residual_grid_std_mean: 0.0262925647128327
- residual_grid_clip_ratio_mean: 4.8955282636597985e-05
- cache_mode: fast_decoded_equivalent_no_entropy_roundtrip
- count: 8192
- crop_size: 512
- detail_downsample_factor: 32
- detail_bits: 4
- detail_range: 0.25
- detail_gain: 1.0
- actual_payload_bpp: 
- paper_bpp: 
- bpp_note: not computed; this cache is for Stage4 training only. Use eval_stage3_uniform_residual_bitstream.py for actual payload bpp.

## Artifacts

- summary: `results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/summary.json`
- per_image_metrics: `results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/per_image_metrics.jsonl`
- reconstruction_manifest: `results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl`
- decoder_feature_cache: `results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/decoder_feature_cache`
- sample: `results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/stage3_training_cache_fast_grid.png`
