# 20260629_stage4_condition_gate_semdrop_detailcontrol_dists_probe600_b2ga2

Date: 2026-06-29T03:03:16

## Command

```bash
scripts/train_stage4_cod_lite_condition_gate.py --adapter-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailcontrol_semdrop020_ft800_b8.pt --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage4_condition_gate_semdrop_detailcontrol_dists_probe600_b2ga2 --crop-size 512 --batch-size 2 --grad-accum-steps 2 --num-workers 2 --max-steps 600 --lr 1e-4 --hidden-channels 128 --num-blocks 2 --gate-min 0.55 --gate-max 1.0 --init-gate 0.85 --condition-l1-weight 0.1 --condition-regression-guard-weight 0.1 --image-l1-weight 0.25 --lpips-weight 0.08 --dists-weight 0.4 --ms-ssim-weight 0.1 --stage3-l1-guard-weight 0.3 --stage3-mse-guard-weight 0.8 --gate-mean-target 0.85 --gate-mean-weight 0.05 --grad-clip-norm 1.0 --wandb-mode offline
```

## Summary

- loss_mean: 0.2278306379293402
- image_l1_mean: 0.05853612515920152
- lpips_mean: 0.4070122229680419
- dists_loss_mean: 0.2752380295594533
- ms_ssim_loss_mean: 0.2780230784416199
- condition_l1_mean: 0.3949366549650828
- ungated_condition_l1_mean: 0.3941270908713341
- condition_regression_guard_mean: 0.0010628861685593922
- stage3_l1_guard_mean: 0.004625707210119193
- stage3_mse_guard_mean: 0.001541900464653736
- gate_mean_loss_mean: 0.010338381802059758
- stage4_psnr_mean: 21.145683142344158
- stage3_psnr_mean: 21.996627297401428
- stage4_ms_ssim_mean: 0.7219769215087096
- stage3_ms_ssim_mean: 0.7440462116897106
- gate_mean_mean: 0.9481961599985759
- gate_std_mean: 0.031232300928192368
- gate_min_mean: 0.843076171875
- gate_max_mean: 0.9888736979166667
- gate_mean_over_steps: 0.9481961599985759
- gate_mean_std_over_steps: 0.024677366018295288
- payload_policy: deterministic decoder-side condition gate from decoded CoSER tensors and fixed model weights; no transmitted side information and no RGB output blending

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_condition_gate/20260629_stage4_condition_gate_semdrop_detailcontrol_dists_probe600_b2ga2.pt`
- summary: `results/stage4_cod_lite_condition_gate/20260629_stage4_condition_gate_semdrop_detailcontrol_dists_probe600_b2ga2/summary.json`
- results_dir: `results/stage4_cod_lite_condition_gate/20260629_stage4_condition_gate_semdrop_detailcontrol_dists_probe600_b2ga2`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_025959-8hbvc7i8`
