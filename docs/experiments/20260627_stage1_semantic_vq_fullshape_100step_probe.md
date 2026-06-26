# 20260627_stage1_semantic_vq_fullshape_100step_probe

Date: 2026-06-27T02:41:09

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage1_semantic_vq.py --disable-lpips --max-steps 100 --batch-size 1 --num-workers 4 --log-every 10 --sample-every 50 --wandb-mode offline --run-name 20260627_stage1_semantic_vq_fullshape_100step_probe
```

## Summary

- loss_total: 0.36492761969566345
- loss_l1_sem: 0.20468799769878387
- loss_ms_ssim_sem: 0.7127061486244202
- loss_vq: 0.011999621987342834
- perplexity: 1.7547653913497925
- dead_code_ratio: 1.0
- used_codes: 2.0
- psnr_sem: 12.800244092941284
- lr: 0.0001
- step: 100

## Artifacts

- checkpoint: `checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_100step_probe.pt`
- output_dir: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_100step_probe`
- summary: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_100step_probe/summary.json`
