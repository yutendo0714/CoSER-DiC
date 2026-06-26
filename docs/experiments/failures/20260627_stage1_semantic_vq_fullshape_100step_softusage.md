# 20260627_stage1_semantic_vq_fullshape_100step_softusage

Date: 2026-06-27T02:42:41

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage1_semantic_vq.py --disable-lpips --max-steps 100 --batch-size 1 --num-workers 4 --log-every 10 --sample-every 50 --wandb-mode offline --run-name 20260627_stage1_semantic_vq_fullshape_100step_softusage
```

## Summary

- loss_total: 0.43496155738830566
- loss_l1_sem: 0.21231357753276825
- loss_ms_ssim_sem: 0.7203091382980347
- loss_vq: 0.03240721672773361
- perplexity: 1.9241317510604858
- soft_perplexity: 1.9909719228744507
- dead_code_ratio: 0.99951171875
- used_codes: 3.0
- usage_loss: 0.9235790371894836
- psnr_sem: 12.487072944641113
- lr: 0.0001
- step: 100

## Artifacts

- checkpoint: `checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_100step_softusage.pt`
- output_dir: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_100step_softusage`
- summary: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_100step_softusage/summary.json`
