# 20260627_stage1_semantic_vq_fullshape_smoke

Date: 2026-06-27T02:40:42

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage1_semantic_vq.py --disable-lpips --max-steps 2 --batch-size 1 --num-workers 0 --log-every 1 --sample-every 2 --wandb-mode offline --run-name 20260627_stage1_semantic_vq_fullshape_smoke
```

## Summary

- loss_total: 0.5054792761802673
- loss_l1_sem: 0.2977506220340729
- loss_ms_ssim_sem: 0.7887020111083984
- loss_vq: 0.04910654574632645
- perplexity: 11.341123580932617
- dead_code_ratio: 0.998046875
- used_codes: 17.0
- psnr_sem: 9.813132882118225
- lr: 0.0001
- step: 2

## Artifacts

- checkpoint: `checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_smoke.pt`
- output_dir: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_smoke`
- summary: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_smoke/summary.json`
