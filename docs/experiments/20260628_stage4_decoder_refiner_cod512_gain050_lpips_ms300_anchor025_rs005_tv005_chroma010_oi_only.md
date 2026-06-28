# 20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms300_anchor025_rs005_tv005_chroma010_oi_only

Date: 2026-06-28T12:43:41

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage4_decoder_refiner.py --config configs/train/train_stage4_decoder_refiner.yaml --run-name 20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms300_anchor025_rs005_tv005_chroma010_oi_only --crop-size 512 --batch-size 2 --max-steps 300 --residual-scale 0.05 --loss-anchor-stage3 0.25 --loss-refiner-tv 0.05 --loss-refiner-chroma 0.10 --sample-every 300 --log-every 25 --wandb-mode offline
```

## Summary

- loss_total: 0.3030441701412201
- loss_l1: 0.07535670697689056
- loss_ms_ssim: 0.36830800771713257
- loss_lpips: 0.6853201389312744
- loss_dists: 0.0
- loss_anchor_stage3: 0.0028324369341135025
- loss_refiner_tv: 0.0036764866672456264
- loss_refiner_chroma: 0.002875934587791562
- psnr_stage3: 19.121129989624023
- psnr_refined: 19.093826293945312
- psnr_delta_refined_vs_stage3: -0.027303695678710938
- l1_stage3: 0.07514151930809021
- l1_refined: 0.07535670697689056
- refiner_residual_abs_mean: 0.0028324127197265625
- lr: 5e-05
- step: 300.0
- grad_norm: 0.20825502276420593
- payload_policy: fixed decoder-side refiner weights; actual_payload_bpp unchanged from Stage 3

## Artifacts

- checkpoint: `checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms300_anchor025_rs005_tv005_chroma010_oi_only.pt`
- output_dir: `outputs/stage4_decoder_refiner/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms300_anchor025_rs005_tv005_chroma010_oi_only`
- summary: `outputs/stage4_decoder_refiner/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms300_anchor025_rs005_tv005_chroma010_oi_only/summary.json`
