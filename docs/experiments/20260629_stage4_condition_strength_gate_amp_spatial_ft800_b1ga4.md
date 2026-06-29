# 20260629_stage4_condition_strength_gate_amp_spatial_ft800_b1ga4

Date: 2026-06-29T13:43:04

## Command

```bash
scripts/train_stage4_cod_lite_condition_gate.py --adapter-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8.pt --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage4_condition_strength_gate_amp_spatial_ft800_b1ga4 --crop-size 512 --batch-size 1 --grad-accum-steps 4 --num-workers 4 --max-steps 800 --lr 8e-5 --hidden-channels 128 --num-blocks 3 --gate-min 0.0 --gate-max 1.5 --init-gate 1.0 --condition-l1-weight 0.04 --condition-regression-guard-weight 0.02 --image-l1-weight 0.18 --lpips-weight 0.11 --dists-weight 0.14 --ms-ssim-weight 0.03 --stage3-l1-guard-weight 0.34 --stage3-mse-guard-weight 0.08 --gate-mean-target 1.08 --gate-mean-weight 0.006 --wandb-mode offline
```

## Summary

- loss_mean: 0.11167629959061742
- image_l1_mean: 0.059064571795461235
- lpips_mean: 0.3702967491111485
- dists_loss_mean: 0.24332242054864764
- ms_ssim_loss_mean: 0.2807138234935701
- condition_l1_mean: 0.397622139044106
- ungated_condition_l1_mean: 0.39511431924067436
- condition_regression_guard_mean: 0.0032872122805565596
- stage3_l1_guard_mean: 0.0049841319938423116
- stage3_mse_guard_mean: 0.0018333626782805368
- gate_mean_loss_mean: 0.002262868523061594
- stage4_psnr_mean: 21.0257273247838
- stage3_psnr_mean: 21.980605925619603
- stage4_ms_ssim_mean: 0.71928617621772
- stage3_ms_ssim_mean: 0.7439190085604787
- gate_mean_mean: 1.1099102401733398
- gate_std_mean: 0.03798065763325212
- gate_min_mean: 1.015806884765625
- gate_max_mean: 1.20300048828125
- gate_mean_over_steps: 1.1099102401733398
- gate_mean_std_over_steps: 0.03295176476240158
- payload_policy: deterministic decoder-side condition gate from decoded CoSER tensors and fixed model weights; no transmitted side information and no RGB output blending

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_condition_gate/20260629_stage4_condition_strength_gate_amp_spatial_ft800_b1ga4.pt`
- summary: `results/stage4_cod_lite_condition_gate/20260629_stage4_condition_strength_gate_amp_spatial_ft800_b1ga4/summary.json`
- results_dir: `results/stage4_cod_lite_condition_gate/20260629_stage4_condition_strength_gate_amp_spatial_ft800_b1ga4`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_133758-p68s8y9b`
