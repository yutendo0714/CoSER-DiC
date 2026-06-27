# 20260627_stage3_residual_ae_d32_c3_b5_r025_2500step_norate_from500_fp32

Date: 2026-06-27T12:11:53

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage3_residual_autoencoder.py --stage1-checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --init-checkpoint checkpoints/stage3_residual_autoencoder/20260627_stage3_residual_ae_d32_c3_b5_r025_500step_norate_fp32.pt --train-root /dpl/open-images-v6/train/data --train-root /dpl/div2k --limit-images 8192 --max-steps 2500 --batch-size 8 --num-workers 8 --log-every 100 --sample-every 1250 --detail-channels 3 --quant-bits 5 --value-range 0.25 --loss-ms-ssim 0.0 --loss-rate-proxy 0.0 --no-mixed-precision --wandb-mode offline --run-name 20260627_stage3_residual_ae_d32_c3_b5_r025_2500step_norate_from500_fp32
```

## Summary

- loss_total: 0.05749830603599548
- loss_l1_aux: 0.05749830603599548
- loss_ms_ssim_aux: 0.0
- loss_rate_proxy: 0.5393712520599365
- psnr_sem: 20.619783401489258
- psnr_aux: 20.861513137817383
- psnr_delta_aux_vs_sem: 0.241729736328125
- l1_sem: 0.060313887894153595
- l1_aux_eval: 0.05728258937597275
- ms_ssim_sem: 0.6914903521537781
- ms_ssim_aux_eval: 0.703509509563446
- h_detail_abs_mean: 0.13484281301498413
- h_detail_std: 0.1532282829284668
- h_detail_clip_ratio: 0.0
- detail_code_entropy_bits: 4.981906414031982
- detail_code_min: 0.0
- detail_code_max: 31.0
- lr: 0.0001
- step: 2500.0
- grad_norm: 0.07233339548110962

## Artifacts

- checkpoint: `checkpoints/stage3_residual_autoencoder/20260627_stage3_residual_ae_d32_c3_b5_r025_2500step_norate_from500_fp32.pt`
- output_dir: `outputs/stage3_residual_autoencoder/20260627_stage3_residual_ae_d32_c3_b5_r025_2500step_norate_from500_fp32`
- summary: `outputs/stage3_residual_autoencoder/20260627_stage3_residual_ae_d32_c3_b5_r025_2500step_norate_from500_fp32/summary.json`
