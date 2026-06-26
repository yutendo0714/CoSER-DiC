# 20260627_stage1_semantic_vq_fullshape_1k_pure_ae_l1_probe

Date: 2026-06-27T03:05:21

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage1_semantic_vq.py --disable-lpips --max-steps 1000 --batch-size 4 --num-workers 4 --log-every 50 --sample-every 250 --wandb-mode offline --fixed-quantize-mix 0 --loss-ms-ssim-sem 0 --loss-vq 0 --loss-codebook-usage 0 --grad-clip-norm 1.0 --no-soft-st --no-normalize-latent --run-name 20260627_stage1_semantic_vq_fullshape_1k_pure_ae_l1_probe
```

## Summary

- loss_total: 0.0846957117319107
- loss_l1_sem: 0.0846957117319107
- loss_ms_ssim_sem: 0.3543904423713684
- loss_vq: 4.984935283660889
- perplexity: 32.81216049194336
- soft_perplexity: 37.04379653930664
- dead_code_ratio: 0.99169921875
- used_codes: 69.0
- usage_loss: 0.5991415977478027
- quantize_mix: 0.0
- vq_scale: 1.0
- usage_scale: 1.0
- psnr_sem: 18.402621746063232
- x_sem_mean: 0.54736328125
- x_sem_std: 0.2425537109375
- x_sem_min: 0.021575927734375
- x_sem_max: 0.97314453125
- x_sem_r_mean: 0.560546875
- x_sem_g_mean: 0.5556640625
- x_sem_b_mean: 0.525390625
- h_mean: 0.036407470703125
- h_std: 2.048828125
- h_absmax: 16.609375
- continuous_std: 2.048828125
- quantized_std: 0.482666015625
- lr: 0.0001
- step: 1000
- grad_norm: 0.7151862978935242

## Artifacts

- checkpoint: `checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_1k_pure_ae_l1_probe.pt`
- output_dir: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_1k_pure_ae_l1_probe`
- summary: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_1k_pure_ae_l1_probe/summary.json`
