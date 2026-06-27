# 20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16

Date: 2026-06-27T23:36:49

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage1_semantic_vq.py --config configs/train/train_stage1_semantic_vq.yaml --init-checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_active5k_lpips002_fp32loss_ft500.pt --fixed-quantize-mix 1 --max-steps 500 --batch-size 16 --num-workers 8 --log-every 25 --sample-every 250 --lr 2e-6 --loss-l1-sem 1.0 --loss-ms-ssim-sem 0 --loss-lpips-sem 0.02 --loss-vq 0.05 --loss-codebook-usage 0.005 --loss-rate-prior 0.0005 --rate-prior-checkpoint checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf384_l6_12kstep_32768tokens_stage1_lpips002_ft500_lr1e4_do02_es.pt --rate-soft-temperature 0.1 --freeze-codebook --force-fp32 --grad-clip-norm 1.0 --wandb-mode offline --run-name 20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16
```

## Summary

- loss_total: 0.16433154046535492
- loss_l1_sem: 0.07444821298122406
- loss_ms_ssim_sem: 0.0
- loss_lpips_sem: 0.6355496644973755
- loss_vq: 1.4111733436584473
- loss_vq_commitment: 1.1289386749267578
- loss_vq_codebook: 1.1289386749267578
- perplexity: 775.3422241210938
- soft_perplexity: 822.1102294921875
- assignment_sample_entropy_bits: 0.1205953061580658
- assignment_avg_entropy_bits: 9.598689079284668
- soft_usage_entropy_bits: 9.683188438415527
- dead_code_ratio: 0.8961181640625
- used_codes: 851.0
- usage_loss: 0.25513941049575806
- quantize_mix: 1.0
- vq_scale: 1.0
- usage_scale: 1.0
- psnr_sem: 18.938746452331543
- x_sem_mean: 0.4500858783721924
- x_sem_std: 0.23863078653812408
- x_sem_min: 0.010598068125545979
- x_sem_max: 0.9975530505180359
- x_sem_r_mean: 0.49514955282211304
- x_sem_g_mean: 0.4529106020927429
- x_sem_b_mean: 0.4021974503993988
- h_mean: 0.02002185955643654
- h_std: 2.464517593383789
- h_absmax: 33.03528594970703
- continuous_std: 2.464517593383789
- quantized_std: 2.2230169773101807
- lr: 2e-06
- step: 500
- loss_rate_prior_bits: 10.67590045928955
- loss_hard_prior_bits: 10.678869247436523
- loss_rate_prior: 0.005337950307875872
- grad_norm: 0.15178312361240387

## Artifacts

- checkpoint: `checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt`
- output_dir: `outputs/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16`
- summary: `outputs/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16/summary.json`
