# 20260628_stage4_detailaware_ft600_condition_gate_probe600_b2

Date: 2026-06-29T00:11:17

## Command

```bash
scripts/train_stage4_cod_lite_condition_gate.py --adapter-checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_detailaware_adapter_idstart_ft600_b2.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_traincache_clic_oi_2048_seed20260628/reconstructions/manifest.jsonl --run-name 20260628_stage4_detailaware_ft600_condition_gate_probe600_b2 --crop-size 512 --batch-size 2 --grad-accum-steps 1 --num-workers 2 --max-steps 600 --lr 5e-5 --hidden-channels 160 --num-blocks 2 --init-gate 0.65 --condition-l1-weight 0.12 --condition-regression-guard-weight 0.25 --image-l1-weight 0.60 --lpips-weight 0.06 --ms-ssim-weight 0.20 --stage3-l1-guard-weight 1.40 --stage3-mse-guard-weight 2.80 --gate-mean-target 0.65 --gate-mean-weight 0.01 --wandb-mode offline
```

## Summary

- loss_mean: 0.16842977079252403
- image_l1_mean: 0.05719512987571458
- lpips_mean: 0.3877706682185332
- ms_ssim_loss_mean: 0.27343404591083526
- condition_l1_mean: 0.37912522102395696
- ungated_condition_l1_mean: 0.3782110168536504
- condition_regression_guard_mean: 0.0009289827446142833
- stage3_l1_guard_mean: 0.004014019465539604
- stage3_mse_guard_mean: 0.0013125516350070636
- gate_mean_loss_mean: 0.1137598752732716
- stage4_psnr_mean: 21.33132544994354
- stage3_psnr_mean: 22.079761603673298
- stage4_ms_ssim_mean: 0.7265659540394942
- stage3_ms_ssim_mean: 0.745745467642943
- gate_mean_mean: 0.9804180908203125
- gate_std_mean: 0.009652798558430126
- gate_min_mean: 0.8611328125
- gate_max_mean: 0.9863216145833333
- gate_mean_over_steps: 0.9804180908203125
- gate_mean_std_over_steps: 0.06770335882902145
- payload_policy: deterministic decoder-side condition gate from decoded CoSER tensors and fixed model weights; no transmitted side information and no RGB output blending

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_condition_gate/20260628_stage4_detailaware_ft600_condition_gate_probe600_b2.pt`
- summary: `results/stage4_cod_lite_condition_gate/20260628_stage4_detailaware_ft600_condition_gate_probe600_b2/summary.json`
- results_dir: `results/stage4_cod_lite_condition_gate/20260628_stage4_detailaware_ft600_condition_gate_probe600_b2`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_000956-p0itsspe`
