# 20260629_stage3_traincache_fast_exactseed_smoke16

Date: 2026-06-29T00:52:33

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/build_stage3_training_cache_fast.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt --detail-prior outputs/stage3_residual_entropy/20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_crop512_8192calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json --image-root /dpl/clic/professional/train --image-root /dpl/clic/mobile/train --image-root /dpl/open-images-v6/train/data --crop-size 512 --max-images 16 --shuffle-images --shuffle-seed 20260628 --batch-size 8 --num-workers 4 --detail-downsample-factor 32 --detail-bits 4 --detail-range 0.25 --detail-gain 1.0 --detail-codec semantic_position_leftctx_huffman --run-name 20260629_stage3_traincache_fast_exactseed_smoke16 --wandb-mode offline
```

## Summary

- semantic_only_psnr_mean: 21.18664562702179
- semantic_only_l1_mean: 0.05833292519673705
- semantic_only_ms_ssim_mean: 0.7310744635760784
- stage3_psnr_mean: 21.683610916137695
- stage3_l1_mean: 0.05527055845595896
- stage3_ms_ssim_mean: 0.7361460588872433
- residual_grid_abs_mean_mean: 0.020804569765459746
- residual_grid_std_mean: 0.027186645660549402
- residual_grid_clip_ratio_mean: 0.0
- cache_mode: fast_decoded_equivalent_no_entropy_roundtrip
- count: 16
- crop_size: 512
- detail_downsample_factor: 32
- detail_bits: 4
- detail_range: 0.25
- detail_gain: 1.0
- actual_payload_bpp: 
- paper_bpp: 
- bpp_note: not computed; this cache is for Stage4 training only. Use eval_stage3_uniform_residual_bitstream.py for actual payload bpp.

## Artifacts

- summary: `results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_exactseed_smoke16/summary.json`
- per_image_metrics: `results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_exactseed_smoke16/per_image_metrics.jsonl`
- reconstruction_manifest: `results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_exactseed_smoke16/reconstructions/manifest.jsonl`
- decoder_feature_cache: `results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_exactseed_smoke16/decoder_feature_cache`
- sample: `results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_exactseed_smoke16/stage3_training_cache_fast_grid.png`
