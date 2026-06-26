# 20260627_stage1_semantic_vq_fullshape_1k_softst

Date: 2026-06-27T02:54:52

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage1_semantic_vq.py --disable-lpips --max-steps 1000 --batch-size 4 --num-workers 4 --log-every 50 --sample-every 250 --quantize-warmup-steps 0 --vq-warmup-steps 0 --usage-warmup-steps 0 --wandb-mode offline --run-name 20260627_stage1_semantic_vq_fullshape_1k_softst
```

## Summary

- loss_total: 0.6922491192817688
- loss_l1_sem: 0.4797399938106537
- loss_ms_ssim_sem: 0.8051323294639587
- loss_vq: 0.0014826684491708875
- perplexity: 1.0
- soft_perplexity: 1.0
- dead_code_ratio: 1.0
- used_codes: 1.0
- usage_loss: 1.0
- quantize_mix: 1.0
- vq_scale: 1.0
- usage_scale: 1.0
- psnr_sem: 5.156959295272827
- lr: 0.0001
- step: 1000

## Artifacts

- checkpoint: `checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_1k_softst.pt`
- output_dir: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_1k_softst`
- summary: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_1k_softst/summary.json`
