# 20260627_stage1_semantic_vq_fullshape_5k_pure_ae_l1

Date: 2026-06-27T03:16:36

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage1_semantic_vq.py --disable-lpips --max-steps 5000 --batch-size 4 --num-workers 4 --log-every 100 --sample-every 500 --wandb-mode offline --fixed-quantize-mix 0 --loss-ms-ssim-sem 0 --loss-vq 0 --loss-codebook-usage 0 --grad-clip-norm 1.0 --no-soft-st --no-normalize-latent --run-name 20260627_stage1_semantic_vq_fullshape_5k_pure_ae_l1
```

## Summary

- loss_total: 0.08689644187688828
- loss_l1_sem: 0.08689644187688828
- loss_ms_ssim_sem: 0.0
- loss_vq: 10.295768737792969
- perplexity: 55.348628997802734
- soft_perplexity: 62.75841522216797
- dead_code_ratio: 0.9873046875
- used_codes: 106.0
- usage_loss: 0.5406356453895569
- quantize_mix: 0.0
- vq_scale: 1.0
- usage_scale: 1.0
- psnr_sem: 17.42783546447754
- x_sem_mean: 0.357421875
- x_sem_std: 0.271484375
- x_sem_min: 0.00894927978515625
- x_sem_max: 0.99755859375
- x_sem_r_mean: 0.4501953125
- x_sem_g_mean: 0.3466796875
- x_sem_b_mean: 0.27490234375
- h_mean: 0.042999267578125
- h_std: 2.93359375
- h_absmax: 34.4375
- continuous_std: 2.93359375
- quantized_std: 0.497802734375
- lr: 0.0001
- step: 5000
- grad_norm: 0.46918803453445435

## Artifacts

- checkpoint: `checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_pure_ae_l1.pt`
- output_dir: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_pure_ae_l1`
- summary: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_pure_ae_l1/summary.json`
