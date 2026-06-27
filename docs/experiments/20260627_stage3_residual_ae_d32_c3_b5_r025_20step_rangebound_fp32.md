# 20260627_stage3_residual_ae_d32_c3_b5_r025_20step_rangebound_fp32

Date: 2026-06-27T12:03:35

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage3_residual_autoencoder.py --stage1-checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --train-root /dpl/open-images-v6/train/data --train-root /dpl/div2k --limit-images 256 --max-steps 20 --batch-size 2 --num-workers 4 --log-every 5 --sample-every 20 --detail-channels 3 --quant-bits 5 --value-range 0.25 --loss-ms-ssim 0.0 --loss-rate-proxy 0.01 --no-mixed-precision --wandb-mode offline --run-name 20260627_stage3_residual_ae_d32_c3_b5_r025_20step_rangebound_fp32
```

## Summary

- loss_total: 0.060297101736068726
- loss_l1_aux: 0.06029321998357773
- loss_ms_ssim_aux: 0.0
- loss_rate_proxy: 0.000388026877772063
- psnr_sem: 21.466449737548828
- psnr_aux: 21.44393539428711
- psnr_delta_aux_vs_sem: -0.02251434326171875
- l1_sem: 0.05989249050617218
- l1_aux_eval: 0.060080960392951965
- ms_ssim_sem: 0.6945637464523315
- ms_ssim_aux_eval: 0.692634105682373
- h_detail_abs_mean: 9.700671944301575e-05
- h_detail_std: 0.00013217971718404442
- h_detail_clip_ratio: 0.0
- detail_code_entropy_bits: 0.9798687696456909
- detail_code_min: 15.0
- detail_code_max: 16.0
- lr: 0.0001
- step: 20.0
- grad_norm: 0.05469929426908493

## Artifacts

- checkpoint: `checkpoints/stage3_residual_autoencoder/20260627_stage3_residual_ae_d32_c3_b5_r025_20step_rangebound_fp32.pt`
- output_dir: `outputs/stage3_residual_autoencoder/20260627_stage3_residual_ae_d32_c3_b5_r025_20step_rangebound_fp32`
- summary: `outputs/stage3_residual_autoencoder/20260627_stage3_residual_ae_d32_c3_b5_r025_20step_rangebound_fp32/summary.json`
