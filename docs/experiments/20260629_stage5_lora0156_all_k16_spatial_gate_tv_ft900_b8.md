# 20260629_stage5_lora0156_all_k16_spatial_gate_tv_ft900_b8

Date: 2026-06-29T20:49:49

## Command

```bash
scripts/train_stage4_cod_lite_condition_gate.py --adapter-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage5_detailtarget_lora0156_denoiser_all_r4_balanced_ft900_b4ga2.pt --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage5_lora0156_all_k16_spatial_gate_tv_ft900_b8 --crop-size 512 --batch-size 8 --num-workers 4 --max-steps 900 --lr 3.0e-5 --hidden-channels 160 --num-blocks 3 --gate-min 0.15 --gate-max 1.05 --init-gate 0.75 --condition-l1-weight 0.04 --condition-regression-guard-weight 0.20 --image-l1-weight 0.22 --lpips-weight 0.28 --dists-weight 0.34 --ms-ssim-weight 0.20 --stage3-l1-guard-weight 0.35 --stage3-mse-guard-weight 0.55 --gate-mean-target 0.78 --gate-mean-weight 0.02 --gate-deviation-weight 0.005 --gate-tv-weight 0.015 --counted-control-mode condition_residual_affine_basis --control-basis outputs/stage5_control_basis/20260629_lora0156_denoiser_all_postaffine_basis_g32s8_k256_train8192_b12/control_basis.pt --control-basis-components 16 --control-basis-candidate-components 256 --control-basis-selection topk --control-basis-range-mode global --control-basis-range-floor 0.000001 --control-codec huffman --control-huffman-key topk_c256_k16_p99_b4_mulaw16 --control-quantizer mu_law --control-mu 16 --control-bits 4 --control-range 0.695025622844696 --control-groups 32 --control-grid-size 8 --control-affine-groups 16 --control-affine-grid-size 1 --control-affine-gain-range 1.0 --control-affine-bias-range 0.25 --control-scale 1.0 --wandb-mode offline
```

## Summary

- loss_mean: 0.2633816670709186
- image_l1_mean: 0.058507044518159496
- lpips_mean: 0.34464378502633836
- dists_loss_mean: 0.23355366205175718
- ms_ssim_loss_mean: 0.2785668526755439
- condition_l1_mean: 0.387928763197528
- ungated_condition_l1_mean: 0.3876661401987076
- pre_control_condition_l1_mean: 0.3935875807205836
- condition_regression_guard_mean: 0.0003203507595592075
- stage3_l1_guard_mean: 0.0044831116093943515
- stage3_mse_guard_mean: 0.0015970716441774534
- gate_mean_loss_mean: 0.027865531824056437
- gate_deviation_loss_mean: 0.02877608985567349
- gate_tv_loss_mean: 0.01055665014414343
- control_payload_bytes_mean: 32.44083333333333
- control_grid_abs_mean_mean: 0.3100896990299225
- stage4_psnr_mean: 21.149836054907905
- stage3_psnr_mean: 21.9689350827535
- stage4_ms_ssim_mean: 0.7214331473244561
- stage3_ms_ssim_mean: 0.7442594745424058
- gate_mean_mean: 0.9426286792755127
- gate_std_mean: 0.029390097685391085
- gate_min_mean: 0.8296397569444445
- gate_max_mean: 1.006640625
- gate_mean_over_steps: 0.9426286792755127
- gate_mean_std_over_steps: 0.03764884173870087
- payload_policy: deterministic decoder-side condition gate from decoded CoSER tensors and fixed model weights; no transmitted gate side information and no RGB output blending
- gate_mean_target: 0.78
- gate_mean_weight: 0.02
- gate_deviation_weight: 0.005
- gate_tv_weight: 0.015
- counted_control_mode: condition_residual_affine_basis
- control_basis: outputs/stage5_control_basis/20260629_lora0156_denoiser_all_postaffine_basis_g32s8_k256_train8192_b12/control_basis.pt
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
- control_range: 0.695025622844696
- control_groups: 32
- control_grid_size: 8
- control_affine_groups: 16
- control_affine_grid_size: 1
- control_affine_gain_range: 1.0
- control_affine_bias_range: 0.25
- control_scale: 1.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_condition_gate/20260629_stage5_lora0156_all_k16_spatial_gate_tv_ft900_b8.pt`
- summary: `results/stage4_cod_lite_condition_gate/20260629_stage5_lora0156_all_k16_spatial_gate_tv_ft900_b8/summary.json`
- results_dir: `results/stage4_cod_lite_condition_gate/20260629_stage5_lora0156_all_k16_spatial_gate_tv_ft900_b8`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_203659-3h4a63vd`
