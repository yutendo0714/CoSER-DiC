# 20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_anchor020_rs005

Date: 2026-06-28T12:01:36

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage4_decoder_refiner.py --config configs/train/train_stage4_decoder_refiner.yaml --run-name 20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_anchor020_rs005 --crop-size 512 --batch-size 2 --max-steps 200 --residual-scale 0.05 --loss-anchor-stage3 0.20 --sample-every 50 --log-every 20 --wandb-mode offline
```

## Summary

- loss_total: 0.2532629370689392
- loss_l1: 0.06597387790679932
- loss_ms_ssim: 0.3690328598022461
- loss_lpips: 0.5290594696998596
- loss_dists: 0.0
- loss_anchor_stage3: 0.00473168445751071
- psnr_stage3: 20.510848999023438
- psnr_refined: 20.43621253967285
- psnr_delta_refined_vs_stage3: -0.07463645935058594
- l1_stage3: 0.06545582413673401
- l1_refined: 0.06597387790679932
- refiner_residual_abs_mean: 0.004730224609375
- lr: 5e-05
- step: 200.0
- grad_norm: 0.22975695133209229
- payload_policy: fixed decoder-side refiner weights; actual_payload_bpp unchanged from Stage 3

## Artifacts

- checkpoint: `checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_anchor020_rs005.pt`
- output_dir: `outputs/stage4_decoder_refiner/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_anchor020_rs005`
- summary: `outputs/stage4_decoder_refiner/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_anchor020_rs005/summary.json`
