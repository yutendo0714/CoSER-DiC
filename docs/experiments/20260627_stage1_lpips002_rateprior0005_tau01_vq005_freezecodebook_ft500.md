# 20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500

Date: 2026-06-27T22:30:57

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage1_semantic_vq.py --config configs/train/train_stage1_semantic_vq.yaml --init-checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_active5k_lpips002_fp32loss_ft500.pt --fixed-quantize-mix 1 --max-steps 500 --batch-size 4 --num-workers 4 --log-every 25 --sample-every 250 --lr 2e-6 --loss-l1-sem 1.0 --loss-ms-ssim-sem 0 --loss-lpips-sem 0.02 --loss-vq 0.05 --loss-codebook-usage 0.005 --loss-rate-prior 0.0005 --rate-prior-checkpoint checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf384_l6_12kstep_32768tokens_stage1_lpips002_ft500_lr1e4_do02_es.pt --rate-soft-temperature 0.1 --freeze-codebook --force-fp32 --grad-clip-norm 1.0 --wandb-mode offline --run-name 20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500
```

## Summary

- loss_total: 0.16179068386554718
- loss_l1_sem: 0.07120159268379211
- loss_ms_ssim_sem: 0.0
- loss_lpips_sem: 0.5904495120048523
- loss_vq: 1.4306681156158447
- loss_vq_commitment: 1.1445344686508179
- loss_vq_codebook: 1.1445344686508179
- perplexity: 212.2247772216797
- soft_perplexity: 227.93011474609375
- assignment_sample_entropy_bits: 0.12484762072563171
- assignment_avg_entropy_bits: 7.729449272155762
- soft_usage_entropy_bits: 7.832447528839111
- dead_code_ratio: 0.97216796875
- used_codes: 228.0
- usage_loss: 0.3975040316581726
- quantize_mix: 1.0
- vq_scale: 1.0
- usage_scale: 1.0
- psnr_sem: 18.7416934967041
- x_sem_mean: 0.3135911226272583
- x_sem_std: 0.2584100365638733
- x_sem_min: 0.0009538833983242512
- x_sem_max: 0.9934551119804382
- x_sem_r_mean: 0.36484581232070923
- x_sem_g_mean: 0.2998585104942322
- x_sem_b_mean: 0.2760690152645111
- h_mean: 0.08552251011133194
- h_std: 3.1631505489349365
- h_absmax: 30.75965690612793
- continuous_std: 3.1631505489349365
- quantized_std: 2.9666190147399902
- lr: 2e-06
- step: 500
- loss_rate_prior_bits: 10.518352508544922
- loss_hard_prior_bits: 10.513205528259277
- loss_rate_prior: 0.005259176716208458
- grad_norm: 0.20846524834632874

## Artifacts

- checkpoint: `checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500.pt`
- output_dir: `outputs/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500`
- summary: `outputs/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500/summary.json`
