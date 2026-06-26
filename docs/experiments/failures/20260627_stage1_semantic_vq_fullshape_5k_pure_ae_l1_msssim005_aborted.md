# 20260627_stage1_semantic_vq_fullshape_5k_pure_ae_l1_msssim005_aborted

Date: 2026-06-27T03:11 JST

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage1_semantic_vq.py --disable-lpips --max-steps 5000 --batch-size 4 --num-workers 4 --log-every 100 --sample-every 500 --wandb-mode offline --fixed-quantize-mix 0 --loss-ms-ssim-sem 0.05 --loss-vq 0 --loss-codebook-usage 0 --grad-clip-norm 1.0 --no-soft-st --no-normalize-latent --run-name 20260627_stage1_semantic_vq_fullshape_5k_pure_ae_l1_msssim005
```

## Outcome

Aborted manually around step 900.

The run started with finite loss but `grad_norm=nan` at step 1. It later drifted
into latent explosion and near-black output:

```text
step ~900
loss_total: 0.542
loss_l1_sem: 0.499
loss_ms_ssim_sem: 0.866
psnr_sem: 5.27 dB
h_std: 8.47
h_absmax: 41.5
x_sem_mean: 0.0034
```

## Diagnosis

MS-SSIM was being computed under CUDA autocast/fp16. The scalar loss stayed
finite, but gradients became non-finite and corrupted the AE warm start.

After forcing MS-SSIM inputs to fp32 and adding a gradient guard, the same
branch still failed immediately:

```text
20260627_stage1_semantic_vq_fullshape_1k_pure_ae_l1_msssim005_fp32ms
FloatingPointError: non-finite gradient norm at step 1: nan
```

## Current Policy

- Do not use MS-SSIM as a Stage 1 training loss for the current Core MVP
  baseline.
- Keep MS-SSIM in analysis/evaluation.
- Raise immediately if clipped gradient norm is non-finite.
- Revisit a stable perceptual/SSIM surrogate after the semantic VQ stream is
  stable.

No checkpoint was intentionally kept from this run.
