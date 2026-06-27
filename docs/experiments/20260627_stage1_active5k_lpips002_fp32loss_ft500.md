# 20260627_stage1_active5k_lpips002_fp32loss_ft500

Date: 2026-06-27T17:57:39

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage1_semantic_vq.py --init-checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --fixed-quantize-mix 1 --loss-l1-sem 1.0 --loss-ms-ssim-sem 0 --loss-lpips-sem 0.02 --loss-vq 0.05 --loss-codebook-usage 0.005 --grad-clip-norm 1.0 --lr 1e-5 --max-steps 500 --batch-size 4 --num-workers 4 --log-every 50 --sample-every 250 --wandb-mode offline --run-name 20260627_stage1_active5k_lpips002_fp32loss_ft500
```

## Summary

- loss_total: 0.14524969458580017
- loss_l1_sem: 0.06800784915685654
- loss_ms_ssim_sem: 0.0
- loss_vq: 1.2485979795455933
- loss_vq_commitment: 0.9988783597946167
- loss_vq_codebook: 0.9988783597946167
- perplexity: 185.009033203125
- soft_perplexity: 195.5888671875
- assignment_sample_entropy_bits: 0.1350250244140625
- assignment_avg_entropy_bits: 7.53145170211792
- soft_usage_entropy_bits: 7.611680507659912
- dead_code_ratio: 0.97412109375
- used_codes: 211.0
- usage_loss: 0.4144861698150635
- quantize_mix: 1.0
- vq_scale: 1.0
- usage_scale: 1.0
- psnr_sem: 20.094757080078125
- x_sem_mean: 0.30615234375
- x_sem_std: 0.1903076171875
- x_sem_min: 0.0125274658203125
- x_sem_max: 0.994140625
- x_sem_r_mean: 0.371337890625
- x_sem_g_mean: 0.30078125
- x_sem_b_mean: 0.24658203125
- h_mean: 0.0723876953125
- h_std: 2.5390625
- h_absmax: 35.0625
- continuous_std: 2.5390625
- quantized_std: 2.318359375
- lr: 1e-05
- step: 500
- grad_norm: 0.24524129927158356

## Artifacts

- checkpoint: `checkpoints/stage1_semantic_vq/20260627_stage1_active5k_lpips002_fp32loss_ft500.pt`
- output_dir: `outputs/stage1_semantic_vq/20260627_stage1_active5k_lpips002_fp32loss_ft500`
- summary: `outputs/stage1_semantic_vq/20260627_stage1_active5k_lpips002_fp32loss_ft500/summary.json`
