# 20260629_stage5_post_control_condition_gate_controlaware_perceptual_smoke20_b4

Date: 2026-06-29T14:41:29

## Command

```bash
scripts/train_stage4_cod_lite_condition_gate.py --adapter-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8.pt --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage5_post_control_condition_gate_controlaware_perceptual_smoke20_b4 --crop-size 512 --batch-size 4 --grad-accum-steps 1 --num-workers 4 --max-steps 20 --lr 5e-5 --hidden-channels 128 --num-blocks 3 --gate-min 0.0 --gate-max 1.40 --init-gate 1.05 --condition-l1-weight 0.07 --condition-regression-guard-weight 0.04 --image-l1-weight 0.20 --lpips-weight 0.11 --dists-weight 0.18 --ms-ssim-weight 0.04 --stage3-l1-guard-weight 0.45 --stage3-mse-guard-weight 0.10 --gate-mean-target 1.05 --gate-mean-weight 0.04 --counted-control-mode condition_residual_affine_basis --control-basis outputs/stage5_control_basis/20260629_detailcontrast_hf_postaffine_basis_g32s8_k256_train8192_b12/control_basis.pt --control-basis-components 32 --control-basis-candidate-components 256 --control-basis-selection topk --control-basis-range-mode global --control-basis-range-floor 0.000001 --control-codec huffman --control-huffman-key topk_c256_k32_p99_b4_mulaw16 --control-quantizer mu_law --control-mu 16 --control-bits 4 --control-range 0.7026548385620117 --control-groups 32 --control-grid-size 8 --control-affine-groups 16 --control-affine-grid-size 1 --control-affine-gain-range 1.0 --control-affine-bias-range 0.25 --control-scale 1.0 --wandb-mode offline
```

## Summary

- loss_mean: 0.13256626278162004
- image_l1_mean: 0.05750593952834606
- lpips_mean: 0.34984137415885924
- dists_loss_mean: 0.23617618083953856
- ms_ssim_loss_mean: 0.26645119190216066
- condition_l1_mean: 0.3847456932067871
- ungated_condition_l1_mean: 0.3840613216161728
- pre_control_condition_l1_mean: 0.39111818969249723
- condition_regression_guard_mean: 0.0006843715906143189
- stage3_l1_guard_mean: 0.005026978696696461
- stage3_mse_guard_mean: 0.0019083544029854238
- gate_mean_loss_mean: 5.357757177648636e-06
- control_payload_bytes_mean: 48.475
- control_grid_abs_mean_mean: 0.253889337182045
- stage4_psnr_mean: 21.24736442565918
- stage3_psnr_mean: 22.242567443847655
- stage4_ms_ssim_mean: 0.7335488080978394
- stage3_ms_ssim_mean: 0.756224399805069
- gate_mean_mean: 1.0489105224609374
- gate_std_mean: 0.0021425582701340316
- gate_min_mean: 1.046875
- gate_max_mean: 1.051953125
- gate_mean_over_steps: 1.0489105224609374
- gate_mean_std_over_steps: 0.0020422779489308596
- payload_policy: deterministic decoder-side condition gate from decoded CoSER tensors and fixed model weights; no transmitted gate side information and no RGB output blending
- counted_control_mode: condition_residual_affine_basis
- control_basis: outputs/stage5_control_basis/20260629_detailcontrast_hf_postaffine_basis_g32s8_k256_train8192_b12/control_basis.pt
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
- control_range: 0.7026548385620117
- control_groups: 32
- control_grid_size: 8
- control_affine_groups: 16
- control_affine_grid_size: 1
- control_affine_gain_range: 1.0
- control_affine_bias_range: 0.25
- control_scale: 1.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_condition_gate/20260629_stage5_post_control_condition_gate_controlaware_perceptual_smoke20_b4.pt`
- summary: `results/stage4_cod_lite_condition_gate/20260629_stage5_post_control_condition_gate_controlaware_perceptual_smoke20_b4/summary.json`
- results_dir: `results/stage4_cod_lite_condition_gate/20260629_stage5_post_control_condition_gate_controlaware_perceptual_smoke20_b4`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_144121-d9e1rrv0`
