# 20260628_stage4_decoder_refiner_guarded_rs003_lpips020_ms025_anchor050_tv010_chroma020_lr5e5_oi_only_1k

Date: 2026-06-28T14:42:41

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage4_decoder_refiner.py --config configs/train/train_stage4_decoder_refiner.yaml --run-name 20260628_stage4_decoder_refiner_guarded_rs003_lpips020_ms025_anchor050_tv010_chroma020_lr5e5_oi_only_1k --crop-size 512 --batch-size 4 --max-steps 1000 --checkpoint-every 250 --residual-scale 0.03 --lr 5e-5 --loss-l1 0.35 --loss-ms-ssim 0.25 --loss-lpips 0.20 --loss-dists 0.00 --loss-anchor-stage3 0.50 --loss-refiner-tv 0.10 --loss-refiner-chroma 0.20 --sample-every 250 --log-every 50 --wandb-mode offline
```

## Summary

- loss_total: 0.18200594186782837
- loss_l1: 0.050135791301727295
- loss_ms_ssim: 0.2652902603149414
- loss_lpips: 0.4787633717060089
- loss_dists: 0.0
- loss_anchor_stage3: 0.0032008918933570385
- loss_refiner_tv: 0.005107174627482891
- loss_refiner_chroma: 0.0013600632082670927
- psnr_stage3: 21.822912216186523
- psnr_refined: 21.77837562561035
- psnr_delta_refined_vs_stage3: -0.044536590576171875
- l1_stage3: 0.04976768419146538
- l1_refined: 0.050135791301727295
- refiner_residual_abs_mean: 0.003200531005859375
- lr: 5e-05
- step: 1000.0
- grad_norm: 0.07805648446083069
- payload_policy: fixed decoder-side refiner weights; actual_payload_bpp unchanged from Stage 3

## CLIC64 Screen

Evaluation command family:
`scripts/eval_stage3_uniform_residual_bitstream.py --eval-protocol cod_reproduction_512 --eval-dataset clic2020_test --max-images 64 --crop-size 512 --compute-perceptual --save-reconstructions`.

Patch FID was computed with `torchmetrics.image.fid.FrechetInceptionDistance`,
GenCodec public-script patch sampling, CLIC patch size 256, split=2 half-shift.
The Stage3 CLIC64 patch256 baseline for this screen is 227.199387.

| checkpoint | actual bpp | dPSNR | dMS-SSIM | dLPIPS | dDISTS | FID patch256 | FID delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| step250 | 0.013593197 | -0.016141 | -0.000618 | -0.009830 | -0.006407 | 228.728561 | +1.529175 |
| step500 | 0.013593197 | -0.037136 | -0.001737 | -0.044482 | -0.017230 | 240.693832 | +13.494446 |
| step750 | 0.013593197 | -0.044784 | -0.002366 | -0.054488 | -0.017597 | 241.990082 | +14.790695 |
| step1000 | 0.013593197 | -0.052518 | -0.002545 | -0.058262 | -0.017781 | 244.556274 | +17.356888 |

Decision: do not promote this guarded refiner. The conservative loss schedule
reduces the distortion drop, especially at step250, but CLIC64 patch256 FID
still regresses for every checkpoint.

## Artifacts

- checkpoint: `checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_guarded_rs003_lpips020_ms025_anchor050_tv010_chroma020_lr5e5_oi_only_1k.pt`
- intermediate_checkpoints: `['checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_guarded_rs003_lpips020_ms025_anchor050_tv010_chroma020_lr5e5_oi_only_1k_step000250.pt', 'checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_guarded_rs003_lpips020_ms025_anchor050_tv010_chroma020_lr5e5_oi_only_1k_step000500.pt', 'checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_guarded_rs003_lpips020_ms025_anchor050_tv010_chroma020_lr5e5_oi_only_1k_step000750.pt']`
- output_dir: `outputs/stage4_decoder_refiner/20260628_stage4_decoder_refiner_guarded_rs003_lpips020_ms025_anchor050_tv010_chroma020_lr5e5_oi_only_1k`
- summary: `outputs/stage4_decoder_refiner/20260628_stage4_decoder_refiner_guarded_rs003_lpips020_ms025_anchor050_tv010_chroma020_lr5e5_oi_only_1k/summary.json`
