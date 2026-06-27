# 20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_lpips002_ft500

Date: 2026-06-27T18:03:44

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/export_stage1_semantic_tokens.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_active5k_lpips002_fp32loss_ft500.pt --max-images 4096 --batch-size 16 --num-workers 8 --wandb-mode offline --run-name 20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_lpips002_ft500
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_active5k_lpips002_fp32loss_ft500.pt
- roots: ['/dpl/race_pilot_openimages_crops', '/dpl/race_pilot_div2k_crops_v3']
- num_images: 256
- crop_size: 256
- codebook_size: 8192
- token_shape: [8, 8]
- total_tokens: 16384
- active_codes: 5363
- global_entropy_bits: 11.98315143585205
- tokens_dtype: torch.int16

## Artifacts

- tokens: `outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_lpips002_ft500/semantic_tokens.pt`
- histograms: `outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_lpips002_ft500/histograms.pt`
- summary: `outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_lpips002_ft500/summary.json`
