# 20260629_stage4_stats_hf_condition_gate_channel_balanced_ft1200_b8_limit64_eval_bppfix

Date: 2026-06-29T11:56:01

## Command

```bash
scripts/eval_stage4_cod_lite_adapter.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailfilm_stats_hf_ft1200_b12.pt --condition-gate-checkpoint checkpoints/stage4_cod_lite_condition_gate/20260629_stage4_stats_hf_condition_gate_channel_balanced_ft1200_b8.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl --per-image-metrics results/stage4_cod_lite_adapter_eval/20260629_stage4_detailfilm_stats_hf_ft1200_b12_limit64_eval/per_image_metrics.jsonl --run-name 20260629_stage4_stats_hf_condition_gate_channel_balanced_ft1200_b8_limit64_eval_bppfix --crop-size 512 --limit 64 --batch-size 4 --num-workers 4 --wandb-mode offline
```

## Summary

- actual_payload_bpp_mean: 0.013654232025146484
- stage3_actual_payload_bpp_mean: 0.013654232025146484
- control_payload_bpp_mean: 0.0
- control_payload_bytes_mean: 0.0
- stage3_psnr_mean: 22.365160256624222
- stage4_psnr_mean: 21.927369117736816
- stage3_ms_ssim_mean: 0.7568007968366146
- stage4_ms_ssim_mean: 0.7469742214307189
- stage3_lpips_alex_mean: 0.5449693585978821
- stage4_lpips_alex_mean: 0.4252813470666297
- stage3_dists_mean: 0.3497471623122692
- stage4_dists_mean: 0.31018055882304907
- stage4_l1_mean: 0.05313683545682579
- stage3_l1_mean: 0.05090900359209627
- condition_l1_mean: 0.4169344617985189
- pre_control_condition_l1_mean: 0.4169344617985189
- control_condition_l1_delta_mean: 0.0
- ungated_condition_l1_mean: 0.4097305843606591
- base_condition_l1_mean: 0.5287581100128591
- condition_l1_delta_vs_base_mean: -0.11182364821434021
- condition_residual_l1_mean: 0.3018910828977823
- condition_delta_raw_l1_mean: 0.5313312369398773
- condition_residual_guard_mean_mean: 1.0
- condition_residual_guard_min_mean: 1.0
- condition_residual_guard_max_mean: 1.0
- control_grid_abs_mean_mean: 0.0
- control_hybrid_mode_index_mean: 0.0
- control_hybrid_rdo_score_mean: 0.0
- stage4_alpha_mean: 1.0
- condition_gate_mean_mean: 0.904547487385571
- count: 64
- stage4_psnr_win_rate: 0.046875
- stage4_ms_ssim_win_rate: 0.21875
- stage4_lpips_win_rate: 0.984375
- stage4_dists_win_rate: 0.921875
- stage4_blend_alpha: 1.0
- stage4_alpha_min: 1.0
- stage4_alpha_max: 1.0
- stage4_alpha_std: 0.0
- stage4_gate_checkpoint: 
- stage4_condition_gate_checkpoint: checkpoints/stage4_cod_lite_condition_gate/20260629_stage4_stats_hf_condition_gate_channel_balanced_ft1200_b8.pt
- condition_residual_scale: 0.85
- condition_residual_tanh: True
- condition_residual_guard: none
- condition_residual_guard_granularity: global
- condition_residual_max_rms_ratio: 0.5
- condition_residual_min_gate: 0.0
- condition_delta_ablation: normal
- counted_control_mode: none
- control_grid_size: 4
- control_groups: 8
- control_dct_coeffs_per_group: 4
- control_affine_groups: 8
- control_affine_grid_size: 4
- control_basis: 
- control_basis_components: 8
- control_basis_candidate_components: 8
- control_basis_selection: prefix
- control_basis_range_mode: global
- control_basis_range_floor: 1e-06
- control_basis_source: 
- control_basis_explained_variance: []
- control_basis_cumulative_explained_variance: []
- control_basis_coefficient_abs_quantiles: {}
- control_basis_coefficient_abs_mean: 0.0
- control_basis_coefficient_abs_std: 0.0
- control_basis_coefficient_abs_max: 0.0
- control_basis_component_abs_p95: []
- control_basis_component_abs_p99: []
- control_codec_type: fixed_bits
- control_huffman_key: 
- control_quantizer: uniform
- control_mu: 16.0
- control_bits: 4
- control_range: 0.25
- control_affine_gain_range: 1.0
- control_affine_bias_range: 0.25
- control_hybrid_selector_bytes: 1
- control_hybrid_selector_bits: 0
- control_hybrid_rate_lambda: 0.0
- control_hybrid_selection_objective: condition_l1
- control_hybrid_fidelity_lambda: 0.0
- control_hybrid_fidelity_metric: image_l1
- control_hybrid_mode_counts: {'none': 64}
- control_scale: 1.0
- control_codec: {'codec': 'uniform_control_grid_fixed_bits', 'bits': 4, 'value_range': 0.25}
- control_basis_codec: {'codec': 'uniform_control_grid_fixed_bits', 'bits': 4, 'value_range': 0.25}
- control_affine_gain_codec: {'codec': 'uniform_control_grid_fixed_bits', 'bits': 4, 'value_range': 1.0}
- control_affine_bias_codec: {'codec': 'uniform_control_grid_fixed_bits', 'bits': 4, 'value_range': 0.25}
- semantic_latent_ablation: normal
- detail_context_ablation: normal
- ablation_shuffle_seed: 1234
- save_reconstruction_kinds: ['reference', 'semantic_only', 'stage3', 'stage4', 'quad']
- stage4_payload_policy: actual_payload_bpp = Stage 3 semantic/detail payload plus any counted control stream payload; fixed CoD-Lite weights, adapter, deterministic decoder-side RGB diagnostic blend, or deterministic condition gate are not image-specific side information

## Artifacts

- summary: `results/stage4_cod_lite_adapter_eval/20260629_stage4_stats_hf_condition_gate_channel_balanced_ft1200_b8_limit64_eval_bppfix/summary.json`
- per_image: `results/stage4_cod_lite_adapter_eval/20260629_stage4_stats_hf_condition_gate_channel_balanced_ft1200_b8_limit64_eval_bppfix/per_image_metrics.jsonl`
- reconstructions: ``
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_115555-sses98gl`
