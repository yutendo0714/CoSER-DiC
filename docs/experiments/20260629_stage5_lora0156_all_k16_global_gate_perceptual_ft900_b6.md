# 20260629_stage5_lora0156_all_k16_global_gate_perceptual_ft900_b6

Date: 2026-06-29T20:01:11

## Command

```bash
scripts/train_stage4_cod_lite_condition_gate.py --adapter-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage5_detailtarget_lora0156_denoiser_all_r4_balanced_ft900_b4ga2.pt --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage5_lora0156_all_k16_global_gate_perceptual_ft900_b6 --crop-size 512 --batch-size 6 --num-workers 4 --max-steps 900 --lr 4.0e-5 --hidden-channels 128 --num-blocks 3 --global-gate --gate-min 0.25 --gate-max 1.05 --init-gate 0.75 --condition-l1-weight 0.04 --condition-regression-guard-weight 0.20 --image-l1-weight 0.22 --lpips-weight 0.28 --dists-weight 0.34 --ms-ssim-weight 0.20 --stage3-l1-guard-weight 0.35 --stage3-mse-guard-weight 0.55 --gate-mean-target 0.75 --gate-mean-weight 0.04 --gate-deviation-weight 0.02 --gate-tv-weight 0.0 --counted-control-mode condition_residual_affine_basis --control-basis outputs/stage5_control_basis/20260629_detailtarget_perceptual_long_postaffine_basis_g32s8_k256_train8192_b12/control_basis.pt --control-basis-components 16 --control-basis-candidate-components 256 --control-basis-selection topk --control-basis-range-mode global --control-basis-range-floor 0.000001 --control-codec huffman --control-huffman-key topk_c256_k16_p99_b4_mulaw16 --control-quantizer mu_law --control-mu 16 --control-bits 4 --control-range 0.6962369680404663 --control-groups 32 --control-grid-size 8 --control-affine-groups 16 --control-affine-grid-size 1 --control-affine-gain-range 1.0 --control-affine-bias-range 0.25 --control-scale 1.0 --wandb-mode offline
```

## Summary

- loss_mean: 0.2659127290050189
- image_l1_mean: 0.058540618734227286
- lpips_mean: 0.3488630504906178
- dists_loss_mean: 0.2359200864036878
- ms_ssim_loss_mean: 0.27810266819265156
- condition_l1_mean: 0.3887709844443533
- ungated_condition_l1_mean: 0.38783645533853106
- pre_control_condition_l1_mean: 0.39373736768960954
- condition_regression_guard_mean: 0.0009352790978219774
- stage3_l1_guard_mean: 0.004315464668519174
- stage3_mse_guard_mean: 0.001538956589388868
- gate_mean_loss_mean: 0.023537624261189636
- gate_deviation_loss_mean: 0.024126841904489486
- gate_tv_loss_mean: 0.0
- control_payload_bytes_mean: 32.46981481481479
- control_grid_abs_mean_mean: 0.31035239617029825
- stage4_psnr_mean: 21.15903733147515
- stage3_psnr_mean: 21.949118429819745
- stage4_ms_ssim_mean: 0.7218973318073485
- stage3_ms_ssim_mean: 0.7435484258996116
- gate_mean_mean: 0.8998553430371814
- gate_std_mean: 0.02162270300478364
- gate_min_mean: 0.8697873263888889
- gate_max_mean: 0.9325347222222222
- gate_mean_over_steps: 0.8998553430371814
- gate_mean_std_over_steps: 0.0328785702586174
- payload_policy: deterministic decoder-side condition gate from decoded CoSER tensors and fixed model weights; no transmitted gate side information and no RGB output blending
- gate_mean_target: 0.75
- gate_mean_weight: 0.04
- gate_deviation_weight: 0.02
- gate_tv_weight: 0.0
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

- checkpoint: `checkpoints/stage4_cod_lite_condition_gate/20260629_stage5_lora0156_all_k16_global_gate_perceptual_ft900_b6.pt`
- summary: `results/stage4_cod_lite_condition_gate/20260629_stage5_lora0156_all_k16_global_gate_perceptual_ft900_b6/summary.json`
- results_dir: `results/stage4_cod_lite_condition_gate/20260629_stage5_lora0156_all_k16_global_gate_perceptual_ft900_b6`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_195125-6crfk2rs`
