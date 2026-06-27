# 20260627_stage1_lpips002_rateprior001_tau01_freezecodebook_ft500

Date: 2026-06-27T22:29:00

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage1_semantic_vq.py --config configs/train/train_stage1_semantic_vq.yaml --init-checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_active5k_lpips002_fp32loss_ft500.pt --max-steps 500 --batch-size 8 --num-workers 4 --log-every 25 --sample-every 250 --lr 5e-6 --loss-lpips-sem 0.002 --loss-codebook-usage 0.02 --loss-rate-prior 0.001 --rate-prior-checkpoint checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf384_l6_12kstep_32768tokens_stage1_lpips002_ft500_lr1e4_do02_es.pt --rate-soft-temperature 0.1 --freeze-codebook --force-fp32 --grad-clip-norm 1.0 --wandb-mode offline --run-name 20260627_stage1_lpips002_rateprior001_tau01_freezecodebook_ft500
```

## Summary

- loss_total: 0.21118782460689545
- loss_l1_sem: 0.05887989327311516
- loss_ms_ssim_sem: 0.0
- loss_lpips_sem: 0.6296031475067139
- loss_vq: 0.1347969025373459
- loss_vq_commitment: 0.10783752053976059
- loss_vq_codebook: 0.10783752053976059
- perplexity: 149.02088928222656
- soft_perplexity: 164.60223388671875
- assignment_sample_entropy_bits: 0.4782291054725647
- assignment_avg_entropy_bits: 7.2193708419799805
- soft_usage_entropy_bits: 7.362840175628662
- dead_code_ratio: 0.971923828125
- used_codes: 230.0
- usage_loss: 0.433627724647522
- quantize_mix: 1.0
- vq_scale: 1.0
- usage_scale: 1.0
- psnr_sem: 20.902655124664307
- x_sem_mean: 0.41743534803390503
- x_sem_std: 0.28254178166389465
- x_sem_min: 0.011360980570316315
- x_sem_max: 0.9993917942047119
- x_sem_r_mean: 0.5008206367492676
- x_sem_g_mean: 0.4014250636100769
- x_sem_b_mean: 0.35006025433540344
- h_mean: -0.06244203820824623
- h_std: 1.238208532333374
- h_absmax: 15.998647689819336
- continuous_std: 1.238208532333374
- quantized_std: 1.230112910270691
- lr: 5e-06
- step: 500
- loss_rate_prior_bits: 7.579265117645264
- loss_hard_prior_bits: 7.582149982452393
- loss_rate_prior: 0.0075792656280100346
- grad_norm: 0.32623714208602905

## Artifacts

- checkpoint: `checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior001_tau01_freezecodebook_ft500.pt`
- output_dir: `outputs/stage1_semantic_vq/20260627_stage1_lpips002_rateprior001_tau01_freezecodebook_ft500`
- summary: `outputs/stage1_semantic_vq/20260627_stage1_lpips002_rateprior001_tau01_freezecodebook_ft500/summary.json`
