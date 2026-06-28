# 20260628_stage4_decoder_refiner_smoke2

Date: 2026-06-28T11:44:50

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage4_decoder_refiner.py --config configs/train/train_stage4_decoder_refiner.yaml --run-name 20260628_stage4_decoder_refiner_smoke2 --crop-size 128 --batch-size 1 --max-steps 2 --limit-images 4 --num-workers 0 --loss-lpips 0 --loss-dists 0 --loss-ms-ssim 0 --loss-l1 1.0 --loss-anchor-stage3 0.05 --sample-every 2 --log-every 1 --wandb-mode offline
```

## Summary

- loss_total: 0.06171882525086403
- loss_l1: 0.06170445680618286
- loss_ms_ssim: 0.0
- loss_lpips: 0.0
- loss_dists: 0.0
- loss_anchor_stage3: 0.00028736391686834395
- psnr_stage3: 21.317546844482422
- psnr_refined: 21.314315795898438
- psnr_delta_refined_vs_stage3: -0.003231048583984375
- l1_stage3: 0.06166611239314079
- l1_refined: 0.06170445680618286
- refiner_residual_abs_mean: 0.0002872943878173828
- lr: 5e-05
- step: 2.0
- grad_norm: 0.08245810866355896
- payload_policy: fixed decoder-side refiner weights; actual_payload_bpp unchanged from Stage 3

## Artifacts

- checkpoint: `checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_smoke2.pt`
- output_dir: `outputs/stage4_decoder_refiner/20260628_stage4_decoder_refiner_smoke2`
- summary: `outputs/stage4_decoder_refiner/20260628_stage4_decoder_refiner_smoke2/summary.json`
