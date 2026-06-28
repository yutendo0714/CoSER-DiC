# 20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_anchor020_rs005_oi_only

Date: 2026-06-28T12:17:39

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage4_decoder_refiner.py --config configs/train/train_stage4_decoder_refiner.yaml --run-name 20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_anchor020_rs005_oi_only --crop-size 512 --batch-size 2 --max-steps 200 --sample-every 200 --log-every 20 --wandb-mode offline
```

## Summary

- loss_total: 0.2863495647907257
- loss_l1: 0.06655348837375641
- loss_ms_ssim: 0.3669746518135071
- loss_lpips: 0.6407694816589355
- loss_dists: 0.0
- loss_anchor_stage3: 0.0037885489873588085
- psnr_stage3: 19.482858657836914
- psnr_refined: 19.448843002319336
- psnr_delta_refined_vs_stage3: -0.034015655517578125
- l1_stage3: 0.0662851333618164
- l1_refined: 0.06655348837375641
- refiner_residual_abs_mean: 0.003787994384765625
- lr: 5e-05
- step: 200.0
- grad_norm: 0.3033581078052521
- payload_policy: fixed decoder-side refiner weights; actual_payload_bpp unchanged from Stage 3

## Artifacts

- checkpoint: `checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_anchor020_rs005_oi_only.pt`
- output_dir: `outputs/stage4_decoder_refiner/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_anchor020_rs005_oi_only`
- summary: `outputs/stage4_decoder_refiner/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_anchor020_rs005_oi_only/summary.json`
