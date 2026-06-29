# 20260629_stage4_detailtarget_perceptual_condition_gate_ft500_b4ga2

Date: 2026-06-29T16:39:59

## Command

```bash
scripts/train_stage4_cod_lite_condition_gate.py --adapter-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailtarget_perceptual_ft700_b8.pt --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage4_detailtarget_perceptual_condition_gate_ft500_b4ga2 --crop-size 512 --batch-size 4 --grad-accum-steps 2 --num-workers 4 --max-steps 500 --lr 5.0e-5 --hidden-channels 128 --num-blocks 3 --gate-min 0.65 --gate-max 1.0 --init-gate 0.9 --condition-l1-weight 0.05 --condition-regression-guard-weight 0.50 --image-l1-weight 0.22 --lpips-weight 0.18 --dists-weight 0.24 --ms-ssim-weight 0.22 --stage3-l1-guard-weight 0.30 --stage3-mse-guard-weight 0.55 --gate-mean-target 0.90 --gate-mean-weight 0.05 --wandb-mode offline
```

## Summary

- loss_mean: 0.22339972159266472
- image_l1_mean: 0.058044914864003656
- lpips_mean: 0.37643630284070967
- dists_loss_mean: 0.24942241978645324
- ms_ssim_loss_mean: 0.2761497222781181
- condition_l1_mean: 0.3938543127179146
- ungated_condition_l1_mean: 0.39379503729939463
- pre_control_condition_l1_mean: 0.39379298186302186
- condition_regression_guard_mean: 7.832565903663635e-05
- stage3_l1_guard_mean: 0.004092123522656039
- stage3_mse_guard_mean: 0.0015130119048699271
- gate_mean_loss_mean: 0.009306268525092292
- control_payload_bytes_mean: 0.0
- control_grid_abs_mean_mean: 0.0
- stage4_psnr_mean: 21.19399200439453
- stage3_psnr_mean: 22.004601976394653
- stage4_ms_ssim_mean: 0.7238502777218818
- stage3_ms_ssim_mean: 0.7448052028417588
- gate_mean_mean: 0.993724594116211
- gate_std_mean: 0.0010669179443939356
- gate_min_mean: 0.974796875
- gate_max_mean: 0.9947109375
- gate_mean_over_steps: 0.993724594116211
- gate_mean_std_over_steps: 0.02284380979835987
- payload_policy: deterministic decoder-side condition gate from decoded CoSER tensors and fixed model weights; no transmitted gate side information and no RGB output blending
- counted_control_mode: none
- control_basis: 
- control_basis_components: 8
- control_basis_candidate_components: 0
- effective_control_basis_candidate_components: 8
- control_basis_selection: prefix
- control_basis_range_mode: global
- control_codec: fixed_bits
- control_huffman_key: 
- control_quantizer: uniform
- control_mu: 16.0
- control_bits: 4
- control_range: 0.25
- control_groups: 8
- control_grid_size: 4
- control_affine_groups: 8
- control_affine_grid_size: 4
- control_affine_gain_range: 1.0
- control_affine_bias_range: 0.25
- control_scale: 1.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_condition_gate/20260629_stage4_detailtarget_perceptual_condition_gate_ft500_b4ga2.pt`
- summary: `results/stage4_cod_lite_condition_gate/20260629_stage4_detailtarget_perceptual_condition_gate_ft500_b4ga2/summary.json`
- results_dir: `results/stage4_cod_lite_condition_gate/20260629_stage4_detailtarget_perceptual_condition_gate_ft500_b4ga2`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_163443-0lwjymbj`
