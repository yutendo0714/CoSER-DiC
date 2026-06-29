# 20260629_stage5_post_control_condition_gate_detailtarget_long_controlaware_perceptual_ft1200_b4

Date: 2026-06-29T17:18:52

## Command

```bash
scripts/train_stage4_cod_lite_condition_gate.py --adapter-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailtarget_perceptual_long_ft1800_b8.pt --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage5_post_control_condition_gate_detailtarget_long_controlaware_perceptual_ft1200_b4 --crop-size 512 --batch-size 4 --num-workers 4 --max-steps 1200 --lr 5.0e-5 --hidden-channels 128 --num-blocks 3 --gate-min 0.0 --gate-max 1.40 --init-gate 1.05 --condition-l1-weight 0.04 --condition-regression-guard-weight 0.35 --image-l1-weight 0.18 --lpips-weight 0.22 --dists-weight 0.30 --ms-ssim-weight 0.16 --stage3-l1-guard-weight 0.24 --stage3-mse-guard-weight 0.42 --gate-mean-target 1.05 --gate-mean-weight 0.03 --counted-control-mode condition_residual_affine_basis --control-basis outputs/stage5_control_basis/20260629_detailtarget_perceptual_long_postaffine_basis_g32s8_k256_train8192_b12/control_basis.pt --control-basis-components 32 --control-basis-candidate-components 256 --control-basis-selection topk --control-basis-range-mode global --control-basis-range-floor 0.000001 --control-codec huffman --control-huffman-key topk_c256_k32_p99_b4_mulaw16 --control-quantizer mu_law --control-mu 16 --control-bits 4 --control-range 0.6962369680404663 --control-groups 32 --control-grid-size 8 --control-affine-groups 16 --control-affine-grid-size 1 --control-affine-gain-range 1.0 --control-affine-bias-range 0.25 --control-scale 1.0 --wandb-mode offline
```

## Summary

- loss_mean: 0.22066882508496444
- image_l1_mean: 0.05858956145743529
- lpips_mean: 0.35115294855088
- dists_loss_mean: 0.2357523584117492
- ms_ssim_loss_mean: 0.27833754921952886
- condition_l1_mean: 0.38792435549199583
- ungated_condition_l1_mean: 0.3872226236263911
- pre_control_condition_l1_mean: 0.39395087386171024
- condition_regression_guard_mean: 0.0007449341565370559
- stage3_l1_guard_mean: 0.004616274016637665
- stage3_mse_guard_mean: 0.0017055068894599875
- gate_mean_loss_mean: 0.0002472859379579025
- control_payload_bytes_mean: 48.7425
- control_grid_abs_mean_mean: 0.25549099975576
- stage4_psnr_mean: 21.11867475827535
- stage3_psnr_mean: 21.982877065340677
- stage4_ms_ssim_mean: 0.7216624507804712
- stage3_ms_ssim_mean: 0.7450588564078013
- gate_mean_mean: 1.0522932012875874
- gate_std_mean: 0.052352753803522015
- gate_min_mean: 0.8935611979166667
- gate_max_mean: 1.2249609375
- gate_mean_over_steps: 1.0522932012875874
- gate_mean_std_over_steps: 0.015557215549051762
- payload_policy: deterministic decoder-side condition gate from decoded CoSER tensors and fixed model weights; no transmitted gate side information and no RGB output blending
- counted_control_mode: condition_residual_affine_basis
- control_basis: outputs/stage5_control_basis/20260629_detailtarget_perceptual_long_postaffine_basis_g32s8_k256_train8192_b12/control_basis.pt
- control_basis_components: 32
- control_basis_candidate_components: 256
- effective_control_basis_candidate_components: 256
- control_basis_selection: topk
- control_basis_range_mode: global
- control_codec: huffman
- control_huffman_key: topk_c256_k32_p99_b4_mulaw16
- control_quantizer: mu_law
- control_mu: 16.0
- control_bits: 4
- control_range: 0.6962369680404663
- control_groups: 32
- control_grid_size: 8
- control_affine_groups: 16
- control_affine_grid_size: 1
- control_affine_gain_range: 1.0
- control_affine_bias_range: 0.25
- control_scale: 1.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_condition_gate/20260629_stage5_post_control_condition_gate_detailtarget_long_controlaware_perceptual_ft1200_b4.pt`
- summary: `results/stage4_cod_lite_condition_gate/20260629_stage5_post_control_condition_gate_detailtarget_long_controlaware_perceptual_ft1200_b4/summary.json`
- results_dir: `results/stage4_cod_lite_condition_gate/20260629_stage5_post_control_condition_gate_detailtarget_long_controlaware_perceptual_ft1200_b4`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_171213-ozob6z3z`
