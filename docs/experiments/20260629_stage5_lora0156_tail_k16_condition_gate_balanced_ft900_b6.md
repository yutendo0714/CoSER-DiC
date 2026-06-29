# 20260629_stage5_lora0156_tail_k16_condition_gate_balanced_ft900_b6

Date: 2026-06-29T19:26:12

## Command

```bash
scripts/train_stage4_cod_lite_condition_gate.py --adapter-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage5_detailtarget_lora0156_denoiser_tail_r4_lr1em05_ft800_b4ga2.pt --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage5_lora0156_tail_k16_condition_gate_balanced_ft900_b6 --crop-size 512 --batch-size 6 --num-workers 4 --max-steps 900 --lr 4.0e-5 --hidden-channels 128 --num-blocks 3 --gate-min 0.0 --gate-max 1.05 --init-gate 0.65 --condition-l1-weight 0.06 --condition-regression-guard-weight 0.45 --image-l1-weight 0.30 --lpips-weight 0.18 --dists-weight 0.22 --ms-ssim-weight 0.30 --stage3-l1-guard-weight 0.55 --stage3-mse-guard-weight 0.85 --gate-mean-target 0.65 --gate-mean-weight 0.08 --gate-deviation-weight 0.04 --gate-tv-weight 0.02 --counted-control-mode condition_residual_affine_basis --control-basis outputs/stage5_control_basis/20260629_lora0156_denoiser_tail_postaffine_basis_g32s8_k256_train8192_b12/control_basis.pt --control-basis-components 16 --control-basis-candidate-components 256 --control-basis-selection topk --control-basis-range-mode global --control-basis-range-floor 0.000001 --control-codec huffman --control-huffman-key topk_c256_k16_p99_b4_mulaw16 --control-quantizer mu_law --control-mu 16 --control-bits 4 --control-range 0.6959795951843262 --control-groups 32 --control-grid-size 8 --control-affine-groups 16 --control-affine-grid-size 1 --control-affine-gain-range 1.0 --control-affine-bias-range 0.25 --control-scale 1.0 --wandb-mode offline
```

## Summary

- loss_mean: 0.25163806236452524
- image_l1_mean: 0.05844812720600102
- lpips_mean: 0.3607493324081103
- dists_loss_mean: 0.2440087569091055
- ms_ssim_loss_mean: 0.2776712152030733
- condition_l1_mean: 0.39134443988402684
- ungated_condition_l1_mean: 0.38714010771777896
- pre_control_condition_l1_mean: 0.3930425199204021
- condition_regression_guard_mean: 0.004204332166247897
- stage3_l1_guard_mean: 0.004437611375728415
- stage3_mse_guard_mean: 0.0015591461008362885
- gate_mean_loss_mean: 0.023285280803241524
- gate_deviation_loss_mean: 0.02439694347223293
- gate_tv_loss_mean: 0.010408562928367044
- control_payload_bytes_mean: 32.45148148148151
- control_grid_abs_mean_mean: 0.30898889266782337
- stage4_psnr_mean: 21.163861598968506
- stage3_psnr_mean: 21.983771262698703
- stage4_ms_ssim_mean: 0.7223287847969267
- stage3_ms_ssim_mean: 0.7446669418281979
- gate_mean_mean: 0.8001476424270206
- gate_std_mean: 0.03215167563693184
- gate_min_mean: 0.7023806423611111
- gate_max_mean: 0.8929991319444445
- gate_mean_over_steps: 0.8001476424270206
- gate_mean_std_over_steps: 0.02722056396305561
- payload_policy: deterministic decoder-side condition gate from decoded CoSER tensors and fixed model weights; no transmitted gate side information and no RGB output blending
- gate_mean_target: 0.65
- gate_mean_weight: 0.08
- gate_deviation_weight: 0.04
- gate_tv_weight: 0.02
- counted_control_mode: condition_residual_affine_basis
- control_basis: outputs/stage5_control_basis/20260629_lora0156_denoiser_tail_postaffine_basis_g32s8_k256_train8192_b12/control_basis.pt
- control_basis_components: 16
- control_basis_candidate_components: 256
- effective_control_basis_candidate_components: 256
- control_basis_selection: topk
- control_basis_range_mode: global
- control_codec: huffman
- control_huffman_key: topk_c256_k16_p99_b4_mulaw16
- control_quantizer: mu_law
- control_mu: 16.0
- control_bits: 4
- control_range: 0.6959795951843262
- control_groups: 32
- control_grid_size: 8
- control_affine_groups: 16
- control_affine_grid_size: 1
- control_affine_gain_range: 1.0
- control_affine_bias_range: 0.25
- control_scale: 1.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_condition_gate/20260629_stage5_lora0156_tail_k16_condition_gate_balanced_ft900_b6.pt`
- summary: `results/stage4_cod_lite_condition_gate/20260629_stage5_lora0156_tail_k16_condition_gate_balanced_ft900_b6/summary.json`
- results_dir: `results/stage4_cod_lite_condition_gate/20260629_stage5_lora0156_tail_k16_condition_gate_balanced_ft900_b6`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_191811-ypa8x071`
