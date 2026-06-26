# 20260627_stage1_semantic_vq_fullshape_1k_warmup500

Date: 2026-06-27T02:52:30

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage1_semantic_vq.py --disable-lpips --max-steps 1000 --batch-size 4 --num-workers 4 --log-every 50 --sample-every 250 --quantize-warmup-steps 500 --vq-warmup-steps 500 --usage-warmup-steps 500 --wandb-mode offline --run-name 20260627_stage1_semantic_vq_fullshape_1k_warmup500
```

## Summary

- loss_total: 0.7454748153686523
- loss_l1_sem: 0.513353705406189
- loss_ms_ssim_sem: 0.8248491287231445
- loss_vq: 0.02394210174679756
- perplexity: 2.4964675903320312
- soft_perplexity: 3.400165319442749
- dead_code_ratio: 0.9990234375
- used_codes: 8.0
- usage_loss: 0.8641842603683472
- quantize_mix: 1.0
- vq_scale: 1.0
- usage_scale: 1.0
- psnr_sem: 4.702750742435455
- lr: 0.0001
- step: 1000

## Artifacts

- checkpoint: `checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_1k_warmup500.pt`
- output_dir: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_1k_warmup500`
- summary: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_1k_warmup500/summary.json`
