# 20260627_stage3_residual_ae_d32_c3_b5_r025_20step_smoke_fp32

Date: 2026-06-27T12:02:48

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage3_residual_autoencoder.py --stage1-checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --train-root /dpl/open-images-v6/train/data --train-root /dpl/div2k --limit-images 256 --max-steps 20 --batch-size 2 --num-workers 4 --log-every 5 --sample-every 20 --detail-channels 3 --quant-bits 5 --value-range 0.25 --loss-ms-ssim 0.0 --loss-rate-proxy 0.01 --no-mixed-precision --wandb-mode offline --run-name 20260627_stage3_residual_ae_d32_c3_b5_r025_20step_smoke_fp32
```

## Summary

- loss_total: 0.06928665190935135
- loss_l1_aux: 0.060295335948467255
- loss_ms_ssim_aux: 0.0
- loss_rate_proxy: 0.8991318941116333
- psnr_sem: 21.466449737548828
- psnr_aux: 21.441238403320312
- psnr_delta_aux_vs_sem: -0.025211334228515625
- l1_sem: 0.05989249050617218
- l1_aux_eval: 0.06008610129356384
- ms_ssim_sem: 0.6945637464523315
- ms_ssim_aux_eval: 0.6926649808883667
- h_detail_abs_mean: 0.22478297352790833
- h_detail_std: 0.3013093173503876
- h_detail_clip_ratio: 0.3489583432674408
- detail_code_entropy_bits: 4.509523391723633
- detail_code_min: 0.0
- detail_code_max: 31.0
- lr: 0.0001
- step: 20.0
- grad_norm: 0.19034212827682495

## Artifacts

- checkpoint: `checkpoints/stage3_residual_autoencoder/20260627_stage3_residual_ae_d32_c3_b5_r025_20step_smoke_fp32.pt`
- output_dir: `outputs/stage3_residual_autoencoder/20260627_stage3_residual_ae_d32_c3_b5_r025_20step_smoke_fp32`
- summary: `outputs/stage3_residual_autoencoder/20260627_stage3_residual_ae_d32_c3_b5_r025_20step_smoke_fp32/summary.json`
