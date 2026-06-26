# 20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only

Date: 2026-06-27T03:57:07

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage1_semantic_vq.py --config configs/train/train_stage1_semantic_vq.yaml --run-name 20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only --max-steps 5000 --batch-size 32 --num-workers 8 --init-checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_pure_ae_l1.pt --no-init-load-codebook --reinit-codebook-from-encoder --codebook-init-max-vectors 131072 --fixed-quantize-mix 1 --loss-ms-ssim-sem 0 --loss-lpips-sem 0 --loss-vq 0.10 --loss-codebook-usage 0.01 --grad-clip-norm 1.0 --lr 5e-5 --no-soft-st --no-normalize-latent --wandb-mode offline
```

## Summary

- loss_total: 0.12944786250591278
- loss_l1_sem: 0.05554176867008209
- loss_ms_ssim_sem: 0.0
- loss_vq: 0.7183747291564941
- perplexity: 1188.2703857421875
- soft_perplexity: 1270.1541748046875
- dead_code_ratio: 0.8291015625
- used_codes: 1400.0
- usage_loss: 0.20686250925064087
- quantize_mix: 1.0
- vq_scale: 1.0
- usage_scale: 1.0
- psnr_sem: 20.518524646759033
- x_sem_mean: 0.46875
- x_sem_std: 0.2734375
- x_sem_min: 0.0004565715789794922
- x_sem_max: 0.99951171875
- x_sem_r_mean: 0.537109375
- x_sem_g_mean: 0.46923828125
- x_sem_b_mean: 0.39990234375
- h_mean: 0.0127410888671875
- h_std: 2.255859375
- h_absmax: 33.9375
- continuous_std: 2.255859375
- quantized_std: 2.150390625
- lr: 5e-05
- step: 5000
- grad_norm: 0.21892733871936798

## Artifacts

- checkpoint: `checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt`
- output_dir: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only`
- summary: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only/summary.json`
