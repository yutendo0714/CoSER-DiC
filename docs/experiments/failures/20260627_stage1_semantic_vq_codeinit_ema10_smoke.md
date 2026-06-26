# 20260627_stage1_semantic_vq_codeinit_ema10_smoke

Date: 2026-06-27T02:44:33

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage1_semantic_vq.py --disable-lpips --max-steps 10 --batch-size 1 --num-workers 0 --log-every 1 --sample-every 10 --wandb-mode offline --run-name 20260627_stage1_semantic_vq_codeinit_ema10_smoke
```

## Summary

- loss_total: 0.4490984380245209
- loss_l1_sem: 0.18296778202056885
- loss_ms_ssim_sem: 0.8525984287261963
- loss_vq: 0.04725443571805954
- perplexity: 1.2633812427520752
- soft_perplexity: 1.3447247743606567
- dead_code_ratio: 1.0
- used_codes: 2.0
- usage_loss: 0.9671299457550049
- psnr_sem: 13.277192115783691
- lr: 0.0001
- step: 10

## Artifacts

- checkpoint: `checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_codeinit_ema10_smoke.pt`
- output_dir: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_codeinit_ema10_smoke`
- summary: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_codeinit_ema10_smoke/summary.json`
