# 20260627_stage1_semantic_vq_debug_smoke_fp32vq

Date: 2026-06-27T02:38:46

## Command

```bash
python scripts/train_stage1_semantic_vq.py --config configs/train/train_stage1_semantic_vq.yaml --max-steps 5 --batch-size 2
```

## Summary

- loss_total: 0.44081076979637146
- loss_l1_sem: 0.26023316383361816
- loss_ms_ssim_sem: 0.8116093277931213
- loss_vq: 0.017163332551717758
- perplexity: 9.154203414916992
- dead_code_ratio: 0.8828125
- used_codes: 15.0
- psnr_sem: 10.283122062683105
- lr: 0.0001
- step: 5

## Artifacts

- checkpoint: `checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_debug_smoke_fp32vq.pt`
- output_dir: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_debug_smoke_fp32vq`
- summary: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_debug_smoke_fp32vq/summary.json`
