# 20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms300_anchor030_rs004_tv020_chroma030_oi_only

Date: 2026-06-28T12:35:04

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage4_decoder_refiner.py --config configs/train/train_stage4_decoder_refiner.yaml --run-name 20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms300_anchor030_rs004_tv020_chroma030_oi_only --crop-size 512 --batch-size 2 --max-steps 300 --residual-scale 0.04 --loss-anchor-stage3 0.30 --loss-refiner-tv 0.20 --loss-refiner-chroma 0.30 --sample-every 300 --log-every 25 --wandb-mode offline
```

## Summary

- loss_total: 0.31424713134765625
- loss_l1: 0.07523874938488007
- loss_ms_ssim: 0.367542028427124
- loss_lpips: 0.7245223522186279
- loss_dists: 0.0
- loss_anchor_stage3: 0.0010356816928833723
- loss_refiner_tv: 0.0011749458499252796
- loss_refiner_chroma: 0.0008822640520520508
- psnr_stage3: 19.12111473083496
- psnr_refined: 19.107467651367188
- psnr_delta_refined_vs_stage3: -0.013647079467773438
- l1_stage3: 0.07514163106679916
- l1_refined: 0.07523874938488007
- refiner_residual_abs_mean: 0.0010356903076171875
- lr: 5e-05
- step: 300.0
- grad_norm: 0.09371517598628998
- payload_policy: fixed decoder-side refiner weights; actual_payload_bpp unchanged from Stage 3

## Artifacts

- checkpoint: `checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms300_anchor030_rs004_tv020_chroma030_oi_only.pt`
- output_dir: `outputs/stage4_decoder_refiner/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms300_anchor030_rs004_tv020_chroma030_oi_only`
- summary: `outputs/stage4_decoder_refiner/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms300_anchor030_rs004_tv020_chroma030_oi_only/summary.json`
