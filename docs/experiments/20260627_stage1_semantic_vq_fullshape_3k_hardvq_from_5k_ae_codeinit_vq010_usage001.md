# 20260627_stage1_semantic_vq_fullshape_3k_hardvq_from_5k_ae_codeinit_vq010_usage001

Date: 2026-06-27T03:29:44

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage1_semantic_vq.py --disable-lpips --max-steps 3000 --batch-size 4 --num-workers 4 --log-every 100 --sample-every 500 --wandb-mode offline --init-checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_pure_ae_l1.pt --no-init-load-codebook --reinit-codebook-from-encoder --codebook-init-max-vectors 131072 --fixed-quantize-mix 1 --loss-ms-ssim-sem 0 --loss-vq 0.10 --loss-codebook-usage 0.01 --grad-clip-norm 1.0 --lr 5e-5 --no-soft-st --no-normalize-latent --run-name 20260627_stage1_semantic_vq_fullshape_3k_hardvq_from_5k_ae_codeinit_vq010_usage001
```

## Summary

- loss_total: 0.10825597494840622
- loss_l1_sem: 0.04722098633646965
- loss_ms_ssim_sem: 0.0
- loss_vq: 0.5684816837310791
- perplexity: 169.7916259765625
- soft_perplexity: 188.33255004882812
- dead_code_ratio: 0.9765625
- used_codes: 192.0
- usage_loss: 0.41868168115615845
- quantize_mix: 1.0
- vq_scale: 1.0
- usage_scale: 1.0
- psnr_sem: 22.820932865142822
- x_sem_mean: 0.311279296875
- x_sem_std: 0.2646484375
- x_sem_min: 0.01751708984375
- x_sem_max: 0.98095703125
- x_sem_r_mean: 0.391845703125
- x_sem_g_mean: 0.33447265625
- x_sem_b_mean: 0.2071533203125
- h_mean: -0.005523681640625
- h_std: 2.318359375
- h_absmax: 24.609375
- continuous_std: 2.318359375
- quantized_std: 2.2421875
- lr: 5e-05
- step: 3000
- grad_norm: 0.524117648601532

## Artifacts

- checkpoint: `checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_3k_hardvq_from_5k_ae_codeinit_vq010_usage001.pt`
- output_dir: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_3k_hardvq_from_5k_ae_codeinit_vq010_usage001`
- summary: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_3k_hardvq_from_5k_ae_codeinit_vq010_usage001/summary.json`
