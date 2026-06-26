# 20260627_stage1_semantic_vq_fullshape_1k_noema

Date: 2026-06-27T02:49:35

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage1_semantic_vq.py --disable-lpips --max-steps 1000 --batch-size 4 --num-workers 4 --log-every 50 --sample-every 250 --wandb-mode offline --run-name 20260627_stage1_semantic_vq_fullshape_1k_noema
```

## Summary

- loss_total: 0.7908127903938293
- loss_l1_sem: 0.45369458198547363
- loss_ms_ssim_sem: 0.728596568107605
- loss_vq: 0.1758982539176941
- perplexity: 30.055068969726562
- soft_perplexity: 501.40582275390625
- dead_code_ratio: 0.99560546875
- used_codes: 36.0
- usage_loss: 0.3100127577781677
- psnr_sem: 5.544324517250061
- lr: 0.0001
- step: 1000

## Artifacts

- checkpoint: `checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_1k_noema.pt`
- output_dir: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_1k_noema`
- summary: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_1k_noema/summary.json`
