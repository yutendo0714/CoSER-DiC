# 20260629_stage4_condition_strength_gate_amp_smoke20_b1ga2

Date: 2026-06-29T13:37:31

## Command

```bash
scripts/train_stage4_cod_lite_condition_gate.py --adapter-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8.pt --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage4_condition_strength_gate_amp_smoke20_b1ga2 --crop-size 512 --batch-size 1 --grad-accum-steps 2 --num-workers 2 --max-steps 20 --lr 8e-5 --hidden-channels 128 --num-blocks 2 --gate-min 0.0 --gate-max 1.5 --init-gate 1.0 --global-gate --condition-l1-weight 0.05 --condition-regression-guard-weight 0.02 --image-l1-weight 0.20 --lpips-weight 0.10 --dists-weight 0.12 --ms-ssim-weight 0.03 --stage3-l1-guard-weight 0.35 --stage3-mse-guard-weight 0.08 --gate-mean-target 1.05 --gate-mean-weight 0.01 --wandb-mode offline
```

## Summary

- loss_mean: 0.10978767182677984
- image_l1_mean: 0.05951963886618614
- lpips_mean: 0.3721292808651924
- dists_loss_mean: 0.25952956825494766
- ms_ssim_loss_mean: 0.2681928738951683
- condition_l1_mean: 0.3974964998662472
- ungated_condition_l1_mean: 0.39746554642915727
- condition_regression_guard_mean: 3.628656268119812e-05
- stage3_l1_guard_mean: 0.004151305602863431
- stage3_mse_guard_mean: 0.0016109554999275133
- gate_mean_loss_mean: 0.00240996852517128
- stage4_psnr_mean: 20.91492567062378
- stage3_psnr_mean: 21.743810725212096
- stage4_ms_ssim_mean: 0.7318071253597737
- stage3_ms_ssim_mean: 0.7494132563471794
- gate_mean_mean: 1.0009765625
- gate_std_mean: 0.0
- gate_min_mean: 1.0009765625
- gate_max_mean: 1.0009765625
- gate_mean_over_steps: 1.0009765625
- gate_mean_std_over_steps: 0.002431621076539159
- payload_policy: deterministic decoder-side condition gate from decoded CoSER tensors and fixed model weights; no transmitted side information and no RGB output blending

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_condition_gate/20260629_stage4_condition_strength_gate_amp_smoke20_b1ga2.pt`
- summary: `results/stage4_cod_lite_condition_gate/20260629_stage4_condition_strength_gate_amp_smoke20_b1ga2/summary.json`
- results_dir: `results/stage4_cod_lite_condition_gate/20260629_stage4_condition_strength_gate_amp_smoke20_b1ga2`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_133726-1dmrtesb`
