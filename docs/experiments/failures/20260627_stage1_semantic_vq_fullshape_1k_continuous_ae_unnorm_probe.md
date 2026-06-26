# 20260627_stage1_semantic_vq_fullshape_1k_continuous_ae_unnorm_probe

Date: 2026-06-27T03:03:02

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage1_semantic_vq.py --disable-lpips --max-steps 1000 --batch-size 4 --num-workers 4 --log-every 50 --sample-every 250 --wandb-mode offline --quantize-warmup-steps 1000000 --vq-warmup-steps 1000000 --usage-warmup-steps 1000000 --no-soft-st --no-normalize-latent --run-name 20260627_stage1_semantic_vq_fullshape_1k_continuous_ae_unnorm_probe
```

## Summary

- loss_total: 0.6737267374992371
- loss_l1_sem: 0.4864882230758667
- loss_ms_ssim_sem: 0.8075389266014099
- loss_vq: 25.68070411682129
- perplexity: 1.0
- soft_perplexity: 1.0
- dead_code_ratio: 1.0
- used_codes: 1.0
- usage_loss: 1.0
- quantize_mix: 0.001
- vq_scale: 0.001
- usage_scale: 0.001
- psnr_sem: 5.0616854429244995
- lr: 0.0001
- step: 1000

## Artifacts

- checkpoint: `checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_1k_continuous_ae_unnorm_probe.pt`
- output_dir: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_1k_continuous_ae_unnorm_probe`
- summary: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_1k_continuous_ae_unnorm_probe/summary.json`
