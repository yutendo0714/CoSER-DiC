# 20260627_stage2_semantic_tokens_32768_oi_div2k_from_stage1_rateprior0005_tau01_vq005_ft500_fullroots

Date: 2026-06-27T23:07:00

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/export_stage1_semantic_tokens.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500.pt --image-root /dpl/open-images-v6/train/data --image-root /dpl/div2k --max-images 32768 --batch-size 16 --num-workers 8 --wandb-mode offline --run-name 20260627_stage2_semantic_tokens_32768_oi_div2k_from_stage1_rateprior0005_tau01_vq005_ft500_fullroots
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500.pt
- roots: ['/dpl/open-images-v6/train/data', '/dpl/div2k']
- num_images: 32768
- crop_size: 256
- codebook_size: 8192
- token_shape: [8, 8]
- total_tokens: 2097152
- active_codes: 8126
- global_entropy_bits: 12.268169403076172
- tokens_dtype: torch.int16

## Artifacts

- tokens: `outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_32768_oi_div2k_from_stage1_rateprior0005_tau01_vq005_ft500_fullroots/semantic_tokens.pt`
- histograms: `outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_32768_oi_div2k_from_stage1_rateprior0005_tau01_vq005_ft500_fullroots/histograms.pt`
- summary: `outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_32768_oi_div2k_from_stage1_rateprior0005_tau01_vq005_ft500_fullroots/summary.json`
