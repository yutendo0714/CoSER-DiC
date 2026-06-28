# 20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms10_fp32loss

Date: 2026-06-28T11:55:10

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage4_decoder_refiner.py --config configs/train/train_stage4_decoder_refiner.yaml --run-name 20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms10_fp32loss --crop-size 512 --batch-size 2 --max-steps 10 --sample-every 10 --log-every 1 --wandb-mode offline
```

## Summary

- loss_total: 0.1864926666021347
- loss_l1: 0.03090379759669304
- loss_ms_ssim: 0.17237770557403564
- loss_lpips: 0.47570914030075073
- loss_dists: 0.0
- loss_anchor_stage3: 0.0006649761344306171
- psnr_stage3: 23.780946731567383
- psnr_refined: 23.778743743896484
- psnr_delta_refined_vs_stage3: -0.0022029876708984375
- l1_stage3: 0.030902862548828125
- l1_refined: 0.03090379759669304
- refiner_residual_abs_mean: 0.00067901611328125
- lr: 5e-05
- step: 10.0
- grad_norm: 0.07593312859535217
- payload_policy: fixed decoder-side refiner weights; actual_payload_bpp unchanged from Stage 3

## Artifacts

- checkpoint: `checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms10_fp32loss.pt`
- output_dir: `outputs/stage4_decoder_refiner/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms10_fp32loss`
- summary: `outputs/stage4_decoder_refiner/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms10_fp32loss/summary.json`
