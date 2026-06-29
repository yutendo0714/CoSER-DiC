# 20260629_stage5_post_control_condition_gate_detailtarget_long_topk16_midguard_ft800_b6

Date: 2026-06-29T18:33:36

## Command

```bash
scripts/train_stage4_cod_lite_condition_gate.py --adapter-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailtarget_perceptual_long_ft1800_b8.pt --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage5_post_control_condition_gate_detailtarget_long_topk16_midguard_ft800_b6 --crop-size 512 --batch-size 6 --num-workers 4 --max-steps 800 --lr 3.5e-5 --hidden-channels 128 --num-blocks 3 --gate-min 0.0 --gate-max 1.25 --init-gate 1.00 --condition-l1-weight 0.09 --condition-regression-guard-weight 0.65 --image-l1-weight 0.30 --lpips-weight 0.15 --dists-weight 0.19 --ms-ssim-weight 0.30 --stage3-l1-guard-weight 0.55 --stage3-mse-guard-weight 0.90 --gate-mean-target 1.00 --gate-mean-weight 0.08 --gate-deviation-weight 0.025 --gate-tv-weight 0.006 --counted-control-mode condition_residual_affine_basis --control-basis outputs/stage5_control_basis/20260629_detailtarget_perceptual_long_postaffine_basis_g32s8_k256_train8192_b12/control_basis.pt --control-basis-components 16 --control-basis-candidate-components 256 --control-basis-selection topk --control-basis-range-mode global --control-basis-range-floor 0.000001 --control-codec huffman --control-huffman-key topk_c256_k16_p99_b4_mulaw16 --control-quantizer mu_law --control-mu 16 --control-bits 4 --control-range 0.6962369680404663 --control-groups 32 --control-grid-size 8 --control-affine-groups 16 --control-affine-grid-size 1 --control-affine-gain-range 1.0 --control-affine-bias-range 0.25 --control-scale 1.0 --wandb-mode offline
```

## Summary

- loss_mean: 0.2387277124635875
- image_l1_mean: 0.05827609563712031
- lpips_mean: 0.3586650412157178
- dists_loss_mean: 0.24075984364375472
- ms_ssim_loss_mean: 0.27649550132453443
- condition_l1_mean: 0.38778776936233045
- ungated_condition_l1_mean: 0.38784865602850915
- pre_control_condition_l1_mean: 0.3939510164782405
- condition_regression_guard_mean: 1.1470876634120941e-05
- stage3_l1_guard_mean: 0.004308252753544366
- stage3_mse_guard_mean: 0.0015980669603595744
- gate_mean_loss_mean: 2.4328780925357842e-05
- gate_deviation_loss_mean: 0.00021718384391551383
- gate_tv_loss_mean: 0.004761666743545448
- control_payload_bytes_mean: 32.43666666666668
- control_grid_abs_mean_mean: 0.309727410338819
- stage4_psnr_mean: 21.160466442108156
- stage3_psnr_mean: 21.982877662181853
- stage4_ms_ssim_mean: 0.7235044986754656
- stage3_ms_ssim_mean: 0.7450588437169791
- gate_mean_mean: 1.0015536151081323
- gate_std_mean: 0.011859633516323811
- gate_min_mean: 0.9713330078125
- gate_max_mean: 1.0377734375
- gate_mean_over_steps: 1.0015536151081323
- gate_mean_std_over_steps: 0.004681352525949478
- payload_policy: deterministic decoder-side condition gate from decoded CoSER tensors and fixed model weights; no transmitted gate side information and no RGB output blending
- gate_mean_target: 1.0
- gate_mean_weight: 0.08
- gate_deviation_weight: 0.025
- gate_tv_weight: 0.006
- counted_control_mode: condition_residual_affine_basis
- control_basis: outputs/stage5_control_basis/20260629_detailtarget_perceptual_long_postaffine_basis_g32s8_k256_train8192_b12/control_basis.pt
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
- control_range: 0.6962369680404663
- control_groups: 32
- control_grid_size: 8
- control_affine_groups: 16
- control_affine_grid_size: 1
- control_affine_gain_range: 1.0
- control_affine_bias_range: 0.25
- control_scale: 1.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_condition_gate/20260629_stage5_post_control_condition_gate_detailtarget_long_topk16_midguard_ft800_b6.pt`
- summary: `results/stage4_cod_lite_condition_gate/20260629_stage5_post_control_condition_gate_detailtarget_long_topk16_midguard_ft800_b6/summary.json`
- results_dir: `results/stage4_cod_lite_condition_gate/20260629_stage5_post_control_condition_gate_detailtarget_long_topk16_midguard_ft800_b6`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_182702-yvgnquwz`
