# 20260628_stage2_semantic_tokens_crop512_b8_probe32_stage1_rateprior0005_b16

Date: 2026-06-28T01:49:40

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/export_stage1_semantic_tokens.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt --image-root /dpl/open-images-v6/validation/data --crop-size 512 --max-images 32 --batch-size 8 --num-workers 8 --wandb-mode offline --run-name 20260628_stage2_semantic_tokens_crop512_b8_probe32_stage1_rateprior0005_b16
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt
- roots: ['/dpl/open-images-v6/validation/data']
- num_images: 32
- crop_size: 512
- codebook_size: 8192
- token_shape: [16, 16]
- total_tokens: 8192
- active_codes: 3690
- global_entropy_bits: 11.345823287963867
- tokens_dtype: torch.int16

## Artifacts

- tokens: `outputs/stage2_semantic_tokens/20260628_stage2_semantic_tokens_crop512_b8_probe32_stage1_rateprior0005_b16/semantic_tokens.pt`
- histograms: `outputs/stage2_semantic_tokens/20260628_stage2_semantic_tokens_crop512_b8_probe32_stage1_rateprior0005_b16/histograms.pt`
- summary: `outputs/stage2_semantic_tokens/20260628_stage2_semantic_tokens_crop512_b8_probe32_stage1_rateprior0005_b16/summary.json`
