# 20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_l1

Date: 2026-06-27T03:20:33

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage1_semantic_vq.py --disable-lpips --max-steps 5000 --batch-size 4 --num-workers 4 --log-every 100 --sample-every 500 --wandb-mode offline --init-checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_pure_ae_l1.pt --no-init-load-codebook --reinit-codebook-from-encoder --codebook-init-max-vectors 131072 --fixed-quantize-mix 1 --loss-ms-ssim-sem 0 --loss-vq 0.25 --loss-codebook-usage 0.02 --grad-clip-norm 1.0 --lr 5e-5 --no-soft-st --no-normalize-latent --run-name 20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_l1
```

## Summary

- loss_total: 0.19003884494304657
- loss_l1_sem: 0.09185317158699036
- loss_ms_ssim_sem: 0.0
- loss_vq: 0.36080241203308105
- perplexity: 193.80479431152344
- soft_perplexity: 224.36534118652344
- dead_code_ratio: 0.97412109375
- used_codes: 211.0
- usage_loss: 0.3992534279823303
- quantize_mix: 1.0
- vq_scale: 1.0
- usage_scale: 1.0
- psnr_sem: 17.438851594924927
- x_sem_mean: 0.42333984375
- x_sem_std: 0.2158203125
- x_sem_min: 0.0027904510498046875
- x_sem_max: 0.986328125
- x_sem_r_mean: 0.450439453125
- x_sem_g_mean: 0.43212890625
- x_sem_b_mean: 0.3876953125
- h_mean: -0.035064697265625
- h_std: 1.3818359375
- h_absmax: 20.265625
- continuous_std: 1.3818359375
- quantized_std: 1.3134765625
- lr: 5e-05
- step: 5000
- grad_norm: 0.37362125515937805

## Artifacts

- checkpoint: `checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_l1.pt`
- output_dir: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_l1`
- summary: `outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_l1/summary.json`
