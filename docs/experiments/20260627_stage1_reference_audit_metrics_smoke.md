# 20260627_stage1_reference_audit_metrics_smoke

Date: 2026-06-27T09:15:45

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage1_semantic_vq.py --debug-small --disable-lpips --max-steps 2 --batch-size 1 --num-workers 0 --log-every 1 --sample-every 2 --wandb-mode offline --run-name 20260627_stage1_reference_audit_metrics_smoke
```

## Summary

- loss_total: 0.591903030872345
- loss_l1_sem: 0.3031468093395233
- loss_ms_ssim_sem: 0.0
- loss_vq: 0.27470171451568604
- loss_vq_commitment: 0.21976137161254883
- loss_vq_codebook: 0.21976137161254883
- perplexity: 12.282657623291016
- soft_perplexity: 32.726173400878906
- assignment_sample_entropy_bits: 3.4248342514038086
- assignment_avg_entropy_bits: 3.6185507774353027
- soft_usage_entropy_bits: 5.032372951507568
- dead_code_ratio: 0.828125
- used_codes: 22.0
- usage_loss: 0.28108954429626465
- quantize_mix: 1.0
- vq_scale: 1.0
- usage_scale: 1.0
- psnr_sem: 9.577962160110474
- x_sem_mean: 0.48486328125
- x_sem_std: 0.0278167724609375
- x_sem_min: 0.321533203125
- x_sem_max: 0.5693359375
- x_sem_r_mean: 0.497802734375
- x_sem_g_mean: 0.50048828125
- x_sem_b_mean: 0.456298828125
- h_mean: -0.01045989990234375
- h_std: 0.296875
- h_absmax: 1.1708984375
- continuous_std: 0.296875
- quantized_std: 0.43115234375
- lr: 0.0001
- step: 2

## Artifacts

- checkpoint: `checkpoints/stage1_semantic_vq/20260627_stage1_reference_audit_metrics_smoke.pt`
- output_dir: `outputs/stage1_semantic_vq/20260627_stage1_reference_audit_metrics_smoke`
- summary: `outputs/stage1_semantic_vq/20260627_stage1_reference_audit_metrics_smoke/summary.json`
