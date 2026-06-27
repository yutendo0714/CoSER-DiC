# 20260627_stage1_rateprior0005_tau01_vq005_b16_smoke40

Date: 2026-06-27T23:35:00

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage1_semantic_vq.py --config configs/train/train_stage1_semantic_vq.yaml --init-checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_active5k_lpips002_fp32loss_ft500.pt --fixed-quantize-mix 1 --max-steps 40 --batch-size 16 --num-workers 8 --log-every 10 --sample-every 1000 --lr 2e-6 --loss-l1-sem 1.0 --loss-ms-ssim-sem 0 --loss-lpips-sem 0.02 --loss-vq 0.05 --loss-codebook-usage 0.005 --loss-rate-prior 0.0005 --rate-prior-checkpoint checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf384_l6_12kstep_32768tokens_stage1_lpips002_ft500_lr1e4_do02_es.pt --rate-soft-temperature 0.1 --freeze-codebook --force-fp32 --grad-clip-norm 1.0 --wandb-mode offline --run-name 20260627_stage1_rateprior0005_tau01_vq005_b16_smoke40
```

## Summary

- loss_total: 0.14863307774066925
- loss_l1_sem: 0.06474326550960541
- loss_ms_ssim_sem: 0.0
- loss_lpips_sem: 0.5380473732948303
- loss_vq: 1.331604242324829
- loss_vq_commitment: 1.0652834177017212
- loss_vq_codebook: 1.0652834177017212
- perplexity: 783.140380859375
- soft_perplexity: 834.4188842773438
- assignment_sample_entropy_bits: 0.11100566387176514
- assignment_avg_entropy_bits: 9.613126754760742
- soft_usage_entropy_bits: 9.704627990722656
- dead_code_ratio: 0.8948974609375
- used_codes: 861.0
- usage_loss: 0.2534902095794678
- quantize_mix: 1.0
- vq_scale: 1.0
- usage_scale: 1.0
- psnr_sem: 19.748494625091553
- x_sem_mean: 0.41525620222091675
- x_sem_std: 0.2823457717895508
- x_sem_min: 0.00022889130923431367
- x_sem_max: 0.9996495246887207
- x_sem_r_mean: 0.4918301999568939
- x_sem_g_mean: 0.42115455865859985
- x_sem_b_mean: 0.33278387784957886
- h_mean: 0.06839950382709503
- h_std: 2.7373671531677246
- h_absmax: 31.587505340576172
- continuous_std: 2.7373671531677246
- quantized_std: 2.5402615070343018
- lr: 2e-06
- step: 40
- loss_rate_prior_bits: 10.562432289123535
- loss_hard_prior_bits: 10.562954902648926
- loss_rate_prior: 0.005281216464936733
- grad_norm: 0.19770629703998566

## Artifacts

- checkpoint: `checkpoints/stage1_semantic_vq/20260627_stage1_rateprior0005_tau01_vq005_b16_smoke40.pt`
- output_dir: `outputs/stage1_semantic_vq/20260627_stage1_rateprior0005_tau01_vq005_b16_smoke40`
- summary: `outputs/stage1_semantic_vq/20260627_stage1_rateprior0005_tau01_vq005_b16_smoke40/summary.json`
