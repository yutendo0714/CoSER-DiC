# 20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_lpips002_ft500_fullroots

Date: 2026-06-27T18:04:43

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/export_stage1_semantic_tokens.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_active5k_lpips002_fp32loss_ft500.pt --image-root /dpl/open-images-v6/train/data --image-root /dpl/div2k --max-images 4096 --batch-size 16 --num-workers 8 --wandb-mode offline --run-name 20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_lpips002_ft500_fullroots
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_active5k_lpips002_fp32loss_ft500.pt
- roots: ['/dpl/open-images-v6/train/data', '/dpl/div2k']
- num_images: 4096
- crop_size: 256
- codebook_size: 8192
- token_shape: [8, 8]
- total_tokens: 262144
- active_codes: 7884
- global_entropy_bits: 12.259037017822266
- tokens_dtype: torch.int16

## Artifacts

- tokens: `outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_lpips002_ft500_fullroots/semantic_tokens.pt`
- histograms: `outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_lpips002_ft500_fullroots/histograms.pt`
- summary: `outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_lpips002_ft500_fullroots/summary.json`
