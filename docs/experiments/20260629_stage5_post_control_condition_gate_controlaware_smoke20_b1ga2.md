# 20260629_stage5_post_control_condition_gate_controlaware_smoke20_b1ga2

Date: 2026-06-29T14:28:56

## Command

```bash
scripts/train_stage4_cod_lite_condition_gate.py --adapter-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8.pt --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage5_post_control_condition_gate_controlaware_smoke20_b1ga2 --crop-size 512 --limit 64 --batch-size 1 --grad-accum-steps 2 --num-workers 2 --max-steps 20 --lr 6e-5 --hidden-channels 128 --num-blocks 3 --gate-min 0.0 --gate-max 1.25 --init-gate 1.0 --condition-l1-weight 0.08 --condition-regression-guard-weight 0.05 --image-l1-weight 0.24 --lpips-weight 0.0 --dists-weight 0.0 --ms-ssim-weight 0.065 --stage3-l1-guard-weight 0.55 --stage3-mse-guard-weight 0.14 --gate-mean-target 1.0 --gate-mean-weight 0.04 --counted-control-mode condition_residual_affine_basis --control-basis outputs/stage5_control_basis/20260629_detailcontrast_hf_postaffine_basis_g32s8_k256_train8192_b12/control_basis.pt --control-basis-components 32 --control-basis-candidate-components 256 --control-basis-selection topk --control-basis-range-mode global --control-basis-range-floor 0.000001 --control-codec huffman --control-huffman-key topk_c256_k32_p99_b4_mulaw16 --control-quantizer mu_law --control-mu 16 --control-bits 4 --control-range 0.7026548385620117 --control-groups 32 --control-grid-size 8 --control-affine-groups 16 --control-affine-grid-size 1 --control-affine-gain-range 1.0 --control-affine-bias-range 0.25 --control-scale 1.0 --wandb-mode offline
```

## Summary

- loss_mean: 0.06920788064599037
- image_l1_mean: 0.06475394451990724
- lpips_mean: 0.0
- dists_loss_mean: 0.0
- ms_ssim_loss_mean: 0.3030270576477051
- condition_l1_mean: 0.3844189651310444
- ungated_condition_l1_mean: 0.384441114962101
- pre_control_condition_l1_mean: 0.39084051325917246
- condition_regression_guard_mean: 6.172060966491699e-06
- stage3_l1_guard_mean: 0.005340155865997076
- stage3_mse_guard_mean: 0.001987049262970686
- gate_mean_loss_mean: 2.6980389520758763e-05
- control_payload_bytes_mean: 48.15
- control_grid_abs_mean_mean: 0.2552695721387863
- stage4_psnr_mean: 20.363360548019408
- stage3_psnr_mean: 21.274516582489014
- stage4_ms_ssim_mean: 0.6969729408621788
- stage3_ms_ssim_mean: 0.7221125692129136
- gate_mean_mean: 0.9965240478515625
- gate_std_mean: 0.0013245376354461768
- gate_min_mean: 0.99267578125
- gate_max_mean: 0.9986328125
- gate_mean_over_steps: 0.9965240478515625
- gate_mean_std_over_steps: 0.0038535227067768574
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

- checkpoint: `checkpoints/stage4_cod_lite_condition_gate/20260629_stage5_post_control_condition_gate_controlaware_smoke20_b1ga2.pt`
- summary: `results/stage4_cod_lite_condition_gate/20260629_stage5_post_control_condition_gate_controlaware_smoke20_b1ga2/summary.json`
- results_dir: `results/stage4_cod_lite_condition_gate/20260629_stage5_post_control_condition_gate_controlaware_smoke20_b1ga2`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_142851-edylyuw8`
