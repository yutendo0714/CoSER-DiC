# 20260627_stage3_residual_ae_d32_c3_b5_r025_500step_norate_fp32

Date: 2026-06-27T12:04:27

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage3_residual_autoencoder.py --stage1-checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --train-root /dpl/open-images-v6/train/data --train-root /dpl/div2k --limit-images 4096 --max-steps 500 --batch-size 4 --num-workers 8 --log-every 25 --sample-every 250 --detail-channels 3 --quant-bits 5 --value-range 0.25 --loss-ms-ssim 0.0 --loss-rate-proxy 0.0 --no-mixed-precision --wandb-mode offline --run-name 20260627_stage3_residual_ae_d32_c3_b5_r025_500step_norate_fp32
```

## Summary

- loss_total: 0.056408584117889404
- loss_l1_aux: 0.056408584117889404
- loss_ms_ssim_aux: 0.0
- loss_rate_proxy: 0.0632241815328598
- psnr_sem: 20.193866729736328
- psnr_aux: 20.208988189697266
- psnr_delta_aux_vs_sem: 0.0151214599609375
- l1_sem: 0.05714762955904007
- l1_aux_eval: 0.05634085088968277
- ms_ssim_sem: 0.739501416683197
- ms_ssim_aux_eval: 0.7405154705047607
- h_detail_abs_mean: 0.01580604538321495
- h_detail_std: 0.0197368822991848
- h_detail_clip_ratio: 0.0
- detail_code_entropy_bits: 2.219637155532837
- detail_code_min: 10.0
- detail_code_max: 18.0
- lr: 0.0001
- step: 500.0
- grad_norm: 0.030776305124163628

## Artifacts

- checkpoint: `checkpoints/stage3_residual_autoencoder/20260627_stage3_residual_ae_d32_c3_b5_r025_500step_norate_fp32.pt`
- output_dir: `outputs/stage3_residual_autoencoder/20260627_stage3_residual_ae_d32_c3_b5_r025_500step_norate_fp32`
- summary: `outputs/stage3_residual_autoencoder/20260627_stage3_residual_ae_d32_c3_b5_r025_500step_norate_fp32/summary.json`
