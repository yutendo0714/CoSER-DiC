# 20260627_stage1_semantic_vq_debug_smoke

Date: 2026-06-27T02:37:54

## Command

```bash
python scripts/train_stage1_semantic_vq.py --config configs/train/train_stage1_semantic_vq.yaml --max-steps 5 --batch-size 2
```

## Summary

- loss_total: nan
- loss_l1_sem: 0.26023322343826294
- loss_ms_ssim_sem: 0.8121493458747864
- loss_vq: 0.017163777723908424
- perplexity: nan
- dead_code_ratio: 0.8828125
- used_codes: 15.0
- psnr_sem: 10.283119678497314
- lr: 0.0001
- step: 5

## Artifacts

- checkpoint: `checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_debug_smoke.pt`
- output_dir: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_debug_smoke`
- summary: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_debug_smoke/summary.json`
