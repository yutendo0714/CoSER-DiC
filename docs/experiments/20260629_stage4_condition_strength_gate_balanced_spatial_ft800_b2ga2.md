# 20260629_stage4_condition_strength_gate_balanced_spatial_ft800_b2ga2

Date: 2026-06-29T13:52:32

## Command

```bash
scripts/train_stage4_cod_lite_condition_gate.py --adapter-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8.pt --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage4_condition_strength_gate_balanced_spatial_ft800_b2ga2 --crop-size 512 --batch-size 2 --grad-accum-steps 2 --num-workers 4 --max-steps 800 --lr 6e-5 --hidden-channels 128 --num-blocks 3 --gate-min 0.0 --gate-max 1.25 --init-gate 1.0 --condition-l1-weight 0.08 --condition-regression-guard-weight 0.04 --image-l1-weight 0.24 --lpips-weight 0.075 --dists-weight 0.10 --ms-ssim-weight 0.065 --stage3-l1-guard-weight 0.55 --stage3-mse-guard-weight 0.14 --gate-mean-target 1.03 --gate-mean-weight 0.045 --wandb-mode offline
```

## Summary

- loss_mean: 0.11977375916205346
- image_l1_mean: 0.058642472160281615
- lpips_mean: 0.37722852604463697
- dists_loss_mean: 0.24914213629439474
- ms_ssim_loss_mean: 0.278391239233315
- condition_l1_mean: 0.39525133404880763
- ungated_condition_l1_mean: 0.395079737175256
- condition_regression_guard_mean: 0.0005071605369448662
- stage3_l1_guard_mean: 0.004569639948604163
- stage3_mse_guard_mean: 0.0016824246598662286
- gate_mean_loss_mean: 0.0001899505188695727
- stage4_psnr_mean: 21.090055146217345
- stage3_psnr_mean: 21.980605914592743
- stage4_ms_ssim_mean: 0.7216087607480586
- stage3_ms_ssim_mean: 0.7439190091565251
- gate_mean_mean: 1.0301551353931426
- gate_std_mean: 0.023355781468635543
- gate_min_mean: 0.9757958984375
- gate_max_mean: 1.090205078125
- gate_mean_over_steps: 1.0301551353931426
- gate_mean_std_over_steps: 0.012607748620212078
- payload_policy: deterministic decoder-side condition gate from decoded CoSER tensors and fixed model weights; no transmitted side information and no RGB output blending

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_condition_gate/20260629_stage4_condition_strength_gate_balanced_spatial_ft800_b2ga2.pt`
- summary: `results/stage4_cod_lite_condition_gate/20260629_stage4_condition_strength_gate_balanced_spatial_ft800_b2ga2/summary.json`
- results_dir: `results/stage4_cod_lite_condition_gate/20260629_stage4_condition_strength_gate_balanced_spatial_ft800_b2ga2`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_134809-snfka7l7`
