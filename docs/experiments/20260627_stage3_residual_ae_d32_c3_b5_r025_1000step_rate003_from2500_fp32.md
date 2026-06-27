# 20260627_stage3_residual_ae_d32_c3_b5_r025_1000step_rate003_from2500_fp32

Date: 2026-06-27T12:16:36

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage3_residual_autoencoder.py --stage1-checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --init-checkpoint checkpoints/stage3_residual_autoencoder/20260627_stage3_residual_ae_d32_c3_b5_r025_2500step_norate_from500_fp32.pt --train-root /dpl/open-images-v6/train/data --train-root /dpl/div2k --limit-images 8192 --max-steps 1000 --batch-size 8 --num-workers 8 --log-every 100 --sample-every 500 --detail-channels 3 --quant-bits 5 --value-range 0.25 --loss-ms-ssim 0.0 --loss-rate-proxy 0.03 --no-mixed-precision --wandb-mode offline --run-name 20260627_stage3_residual_ae_d32_c3_b5_r025_1000step_rate003_from2500_fp32
```

## Summary

- loss_total: 0.05410239100456238
- loss_l1_aux: 0.05405929684638977
- loss_ms_ssim_aux: 0.0
- loss_rate_proxy: 0.0014364745002239943
- psnr_sem: 19.223098754882812
- psnr_aux: 19.267187118530273
- psnr_delta_aux_vs_sem: 0.04408836364746094
- l1_sem: 0.055567484349012375
- l1_aux_eval: 0.05391867086291313
- ms_ssim_sem: 0.7703810930252075
- ms_ssim_aux_eval: 0.7801694273948669
- h_detail_abs_mean: 0.00035911862505599856
- h_detail_std: 0.00045741148642264307
- h_detail_clip_ratio: 0.0
- detail_code_entropy_bits: 0.9921586513519287
- detail_code_min: 15.0
- detail_code_max: 16.0
- lr: 0.0001
- step: 1000.0
- grad_norm: 0.05407501757144928

## Artifacts

- checkpoint: `checkpoints/stage3_residual_autoencoder/20260627_stage3_residual_ae_d32_c3_b5_r025_1000step_rate003_from2500_fp32.pt`
- output_dir: `outputs/stage3_residual_autoencoder/20260627_stage3_residual_ae_d32_c3_b5_r025_1000step_rate003_from2500_fp32`
- summary: `outputs/stage3_residual_autoencoder/20260627_stage3_residual_ae_d32_c3_b5_r025_1000step_rate003_from2500_fp32/summary.json`
