# 20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_fp32loss

Date: 2026-06-28T11:56:18

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage4_decoder_refiner.py --config configs/train/train_stage4_decoder_refiner.yaml --run-name 20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_fp32loss --crop-size 512 --batch-size 2 --max-steps 200 --sample-every 50 --log-every 20 --wandb-mode offline
```

## Summary

- loss_total: 0.2503242492675781
- loss_l1: 0.06627771258354187
- loss_ms_ssim: 0.369653582572937
- loss_lpips: 0.5204147100448608
- loss_dists: 0.0
- loss_anchor_stage3: 0.007716119289398193
- psnr_stage3: 20.510860443115234
- psnr_refined: 20.36533546447754
- psnr_delta_refined_vs_stage3: -0.1455249786376953
- l1_stage3: 0.06545577943325043
- l1_refined: 0.06627771258354187
- refiner_residual_abs_mean: 0.007717132568359375
- lr: 5e-05
- step: 200.0
- grad_norm: 0.2619283199310303
- payload_policy: fixed decoder-side refiner weights; actual_payload_bpp unchanged from Stage 3

## Artifacts

- checkpoint: `checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_fp32loss.pt`
- output_dir: `outputs/stage4_decoder_refiner/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_fp32loss`
- summary: `outputs/stage4_decoder_refiner/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms200_fp32loss/summary.json`
