# 20260627_stage1_semantic_vq_fullshape_1k_continuous_ae_probe

Date: 2026-06-27T02:59:55

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage1_semantic_vq.py --disable-lpips --max-steps 1000 --batch-size 4 --num-workers 4 --log-every 50 --sample-every 250 --quantize-warmup-steps 1000000 --vq-warmup-steps 1000000 --usage-warmup-steps 1000000 --wandb-mode offline --run-name 20260627_stage1_semantic_vq_fullshape_1k_continuous_ae_probe
```

## Summary

- loss_total: 0.6784475445747375
- loss_l1_sem: 0.5132439732551575
- loss_ms_ssim_sem: 0.8248440027236938
- loss_vq: 1.931586503982544
- perplexity: 2.747647523880005
- soft_perplexity: 3871.95458984375
- dead_code_ratio: 0.99951171875
- used_codes: 4.0
- usage_loss: 0.08316570520401001
- quantize_mix: 0.001
- vq_scale: 0.001
- usage_scale: 0.001
- psnr_sem: 4.704310894012451
- lr: 0.0001
- step: 1000

## Artifacts

- checkpoint: `checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_1k_continuous_ae_probe.pt`
- output_dir: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_1k_continuous_ae_probe`
- summary: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_1k_continuous_ae_probe/summary.json`
