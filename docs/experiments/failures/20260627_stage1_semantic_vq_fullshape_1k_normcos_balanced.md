# 20260627_stage1_semantic_vq_fullshape_1k_normcos_balanced

Date: 2026-06-27T02:58:44

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage1_semantic_vq.py --disable-lpips --max-steps 1000 --batch-size 4 --num-workers 4 --log-every 50 --sample-every 250 --quantize-warmup-steps 0 --vq-warmup-steps 0 --usage-warmup-steps 0 --wandb-mode offline --run-name 20260627_stage1_semantic_vq_fullshape_1k_normcos_balanced
```

## Summary

- loss_total: 0.9242885112762451
- loss_l1_sem: 0.5461472868919373
- loss_ms_ssim_sem: 0.9037659764289856
- loss_vq: 1.8583123683929443
- perplexity: 9.676498413085938
- soft_perplexity: 6651.77490234375
- dead_code_ratio: 0.99658203125
- used_codes: 29.0
- usage_loss: 0.023113608360290527
- quantize_mix: 1.0
- vq_scale: 1.0
- usage_scale: 1.0
- psnr_sem: 4.3013182282447815
- lr: 0.0001
- step: 1000

## Artifacts

- checkpoint: `checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_1k_normcos_balanced.pt`
- output_dir: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_1k_normcos_balanced`
- summary: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_1k_normcos_balanced/summary.json`
