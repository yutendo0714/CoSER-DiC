# 20260628_stage4_decoder_refiner_checkpoint_every_smoke2

Date: 2026-06-28T13:20:21

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage4_decoder_refiner.py --config configs/train/train_stage4_decoder_refiner.yaml --run-name 20260628_stage4_decoder_refiner_checkpoint_every_smoke2 --crop-size 128 --batch-size 1 --max-steps 2 --checkpoint-every 1 --limit-images 4 --loss-ms-ssim 0 --loss-lpips 0 --loss-dists 0 --sample-every 2 --log-every 1 --wandb-mode offline
```

## Summary

- loss_total: 0.02333897538483143
- loss_l1: 0.07768850773572922
- loss_ms_ssim: 0.0
- loss_lpips: 0.0
- loss_dists: 0.0
- loss_anchor_stage3: 9.64410719461739e-05
- loss_refiner_tv: 1.0080644642584957e-05
- loss_refiner_chroma: 7.808434020262212e-05
- psnr_stage3: 18.859437942504883
- psnr_refined: 18.859134674072266
- psnr_delta_refined_vs_stage3: -0.0003032684326171875
- l1_stage3: 0.07768264412879944
- l1_refined: 0.07768850773572922
- refiner_residual_abs_mean: 9.644031524658203e-05
- lr: 5e-05
- step: 2.0
- grad_norm: 0.04218568280339241
- payload_policy: fixed decoder-side refiner weights; actual_payload_bpp unchanged from Stage 3

## Artifacts

- checkpoint: `checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_checkpoint_every_smoke2.pt`
- intermediate_checkpoints: `['checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_checkpoint_every_smoke2_step000001.pt']`
- output_dir: `outputs/stage4_decoder_refiner/20260628_stage4_decoder_refiner_checkpoint_every_smoke2`
- summary: `outputs/stage4_decoder_refiner/20260628_stage4_decoder_refiner_checkpoint_every_smoke2/summary.json`
