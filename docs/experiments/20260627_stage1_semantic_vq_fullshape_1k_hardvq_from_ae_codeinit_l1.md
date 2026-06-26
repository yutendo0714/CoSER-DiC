# 20260627_stage1_semantic_vq_fullshape_1k_hardvq_from_ae_codeinit_l1

Date: 2026-06-27T03:08:18

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage1_semantic_vq.py --disable-lpips --max-steps 1000 --batch-size 4 --num-workers 4 --log-every 50 --sample-every 250 --wandb-mode offline --init-checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_1k_pure_ae_l1_probe.pt --no-init-load-codebook --reinit-codebook-from-encoder --codebook-init-max-vectors 65536 --fixed-quantize-mix 1 --loss-ms-ssim-sem 0 --loss-vq 0.25 --loss-codebook-usage 0.01 --grad-clip-norm 1.0 --lr 5e-5 --no-soft-st --no-normalize-latent --run-name 20260627_stage1_semantic_vq_fullshape_1k_hardvq_from_ae_codeinit_l1
```

## Summary

- loss_total: 0.11296417564153671
- loss_l1_sem: 0.06691309809684753
- loss_ms_ssim_sem: 0.2956240177154541
- loss_vq: 0.16702696681022644
- perplexity: 141.056640625
- soft_perplexity: 170.94163513183594
- dead_code_ratio: 0.9775390625
- used_codes: 183.0
- usage_loss: 0.4294338822364807
- quantize_mix: 1.0
- vq_scale: 1.0
- usage_scale: 1.0
- psnr_sem: 19.28715229034424
- x_sem_mean: 0.297119140625
- x_sem_std: 0.28271484375
- x_sem_min: 0.01024627685546875
- x_sem_max: 0.99853515625
- x_sem_r_mean: 0.364501953125
- x_sem_g_mean: 0.308837890625
- x_sem_b_mean: 0.218017578125
- h_mean: 0.01251220703125
- h_std: 1.994140625
- h_absmax: 11.859375
- continuous_std: 1.994140625
- quantized_std: 1.986328125
- lr: 5e-05
- step: 1000
- grad_norm: 0.33443859219551086

## Artifacts

- checkpoint: `checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_1k_hardvq_from_ae_codeinit_l1.pt`
- output_dir: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_1k_hardvq_from_ae_codeinit_l1`
- summary: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_1k_hardvq_from_ae_codeinit_l1/summary.json`
