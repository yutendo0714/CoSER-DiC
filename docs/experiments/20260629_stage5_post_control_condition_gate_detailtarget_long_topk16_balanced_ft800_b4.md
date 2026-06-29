# 20260629_stage5_post_control_condition_gate_detailtarget_long_topk16_balanced_ft800_b4

Date: 2026-06-29T18:10:58

## Command

```bash
scripts/train_stage4_cod_lite_condition_gate.py --adapter-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailtarget_perceptual_long_ft1800_b8.pt --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage5_post_control_condition_gate_detailtarget_long_topk16_balanced_ft800_b4 --crop-size 512 --batch-size 4 --num-workers 4 --max-steps 800 --lr 4.0e-5 --hidden-channels 128 --num-blocks 3 --gate-min 0.0 --gate-max 1.25 --init-gate 1.00 --condition-l1-weight 0.08 --condition-regression-guard-weight 0.55 --image-l1-weight 0.28 --lpips-weight 0.16 --dists-weight 0.20 --ms-ssim-weight 0.28 --stage3-l1-guard-weight 0.45 --stage3-mse-guard-weight 0.75 --gate-mean-target 1.00 --gate-mean-weight 0.08 --counted-control-mode condition_residual_affine_basis --control-basis outputs/stage5_control_basis/20260629_detailtarget_perceptual_long_postaffine_basis_g32s8_k256_train8192_b12/control_basis.pt --control-basis-components 16 --control-basis-candidate-components 256 --control-basis-selection topk --control-basis-range-mode global --control-basis-range-floor 0.000001 --control-codec huffman --control-huffman-key topk_c256_k16_p99_b4_mulaw16 --control-quantizer mu_law --control-mu 16 --control-bits 4 --control-range 0.6962369680404663 --control-groups 32 --control-grid-size 8 --control-affine-groups 16 --control-affine-grid-size 1 --control-affine-gain-range 1.0 --control-affine-bias-range 0.25 --control-scale 1.0 --wandb-mode offline
```

## Summary

- loss_mean: 0.23403030868619681
- image_l1_mean: 0.058456531285773966
- lpips_mean: 0.3590849014371634
- dists_loss_mean: 0.2401037084311247
- ms_ssim_loss_mean: 0.27828880183398724
- condition_l1_mean: 0.3881280303373933
- ungated_condition_l1_mean: 0.38818653386086227
- pre_control_condition_l1_mean: 0.3941863204166293
- condition_regression_guard_mean: 4.027493298053741e-05
- stage3_l1_guard_mean: 0.004394830546225421
- stage3_mse_guard_mean: 0.0016149935705834651
- gate_mean_loss_mean: 7.471675349279394e-05
- control_payload_bytes_mean: 32.4534375
- control_grid_abs_mean_mean: 0.3090021272189915
- stage4_psnr_mean: 21.14758095741272
- stage3_psnr_mean: 21.98060589790344
- stage4_ms_ssim_mean: 0.7217111981660128
- stage3_ms_ssim_mean: 0.7439190078526735
- gate_mean_mean: 1.005929993391037
- gate_std_mean: 0.02321431510594266
- gate_min_mean: 0.9339697265625
- gate_max_mean: 1.074228515625
- gate_mean_over_steps: 1.005929993391037
- gate_mean_std_over_steps: 0.006289032753556967
- payload_policy: deterministic decoder-side condition gate from decoded CoSER tensors and fixed model weights; no transmitted gate side information and no RGB output blending
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

- checkpoint: `checkpoints/stage4_cod_lite_condition_gate/20260629_stage5_post_control_condition_gate_detailtarget_long_topk16_balanced_ft800_b4.pt`
- summary: `results/stage4_cod_lite_condition_gate/20260629_stage5_post_control_condition_gate_detailtarget_long_topk16_balanced_ft800_b4/summary.json`
- results_dir: `results/stage4_cod_lite_condition_gate/20260629_stage5_post_control_condition_gate_detailtarget_long_topk16_balanced_ft800_b4`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_180633-snqvfcd8`
