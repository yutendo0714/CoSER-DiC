# 20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b8

Date: 2026-06-27T23:38:33

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage1_semantic_vq.py --config configs/train/train_stage1_semantic_vq.yaml --init-checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_active5k_lpips002_fp32loss_ft500.pt --fixed-quantize-mix 1 --max-steps 500 --batch-size 8 --num-workers 8 --log-every 25 --sample-every 250 --lr 2e-6 --loss-l1-sem 1.0 --loss-ms-ssim-sem 0 --loss-lpips-sem 0.02 --loss-vq 0.05 --loss-codebook-usage 0.005 --loss-rate-prior 0.0005 --rate-prior-checkpoint checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf384_l6_12kstep_32768tokens_stage1_lpips002_ft500_lr1e4_do02_es.pt --rate-soft-temperature 0.1 --freeze-codebook --force-fp32 --grad-clip-norm 1.0 --wandb-mode offline --run-name 20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b8
```

## Summary

- loss_total: 0.1170150637626648
- loss_l1_sem: 0.04771912842988968
- loss_ms_ssim_sem: 0.0
- loss_lpips_sem: 0.5321422815322876
- loss_vq: 1.0409526824951172
- loss_vq_commitment: 0.8327621221542358
- loss_vq_codebook: 0.8327621221542358
- perplexity: 359.9085693359375
- soft_perplexity: 392.2930603027344
- assignment_sample_entropy_bits: 0.15502144396305084
- assignment_avg_entropy_bits: 8.491486549377441
- soft_usage_entropy_bits: 8.615788459777832
- dead_code_ratio: 0.94970703125
- used_codes: 412.0
- usage_loss: 0.3372471332550049
- quantize_mix: 1.0
- vq_scale: 1.0
- usage_scale: 1.0
- psnr_sem: 22.18200922012329
- x_sem_mean: 0.3888852596282959
- x_sem_std: 0.27772122621536255
- x_sem_min: 0.005653546191751957
- x_sem_max: 0.9995415210723877
- x_sem_r_mean: 0.5111582279205322
- x_sem_g_mean: 0.35736095905303955
- x_sem_b_mean: 0.2981366515159607
- h_mean: 0.04156987741589546
- h_std: 2.714292526245117
- h_absmax: 31.660335540771484
- continuous_std: 2.714292526245117
- quantized_std: 2.5389649868011475
- lr: 2e-06
- step: 500
- loss_rate_prior_bits: 9.8384370803833
- loss_hard_prior_bits: 9.830533027648926
- loss_rate_prior: 0.004919218830764294
- grad_norm: 0.27277183532714844

## Artifacts

- checkpoint: `checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b8.pt`
- output_dir: `outputs/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b8`
- summary: `outputs/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b8/summary.json`
