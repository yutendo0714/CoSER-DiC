# 20260627_stage1_semantic_vq_fullshape_100step_noema

Date: 2026-06-27T02:45:13

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage1_semantic_vq.py --no-ema --disable-lpips --max-steps 100 --batch-size 1 --num-workers 4 --log-every 10 --sample-every 50 --wandb-mode offline --run-name 20260627_stage1_semantic_vq_fullshape_100step_noema
```

## Summary

- loss_total: 0.8580228090286255
- loss_l1_sem: 0.23121261596679688
- loss_ms_ssim_sem: 0.6956761479377747
- loss_vq: 0.4524255394935608
- perplexity: 9.469864845275879
- soft_perplexity: 14.272448539733887
- dead_code_ratio: 0.9970703125
- used_codes: 24.0
- usage_loss: 0.704987645149231
- psnr_sem: 11.558828353881836
- lr: 0.0001
- step: 100

## Artifacts

- checkpoint: `checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_100step_noema.pt`
- output_dir: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_100step_noema`
- summary: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_100step_noema/summary.json`
