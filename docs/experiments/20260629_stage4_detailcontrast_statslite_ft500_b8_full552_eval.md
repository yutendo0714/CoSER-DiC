# 20260629_stage4_detailcontrast_statslite_ft500_b8_full552_eval

Date: 2026-06-29T15:56:52

## Command

```bash
scripts/eval_stage4_cod_lite_adapter.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailcontrast_statslite_ft500_b8.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl --per-image-metrics results/stage4_cod_lite_adapter_eval/20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8_full552_eval/per_image_metrics.jsonl --run-name 20260629_stage4_detailcontrast_statslite_ft500_b8_full552_eval --crop-size 512 --batch-size 8 --num-workers 4 --blend-alpha 1.0 --wandb-mode offline
```

## Summary

- actual_payload_bpp_mean: 0.013999109682829483
- stage3_actual_payload_bpp_mean: 0.013999109682829483
- control_payload_bpp_mean: 0.0
- control_payload_bytes_mean: 0.0
- stage3_psnr_mean: 21.99508854271709
- stage4_psnr_mean: 21.28228445675062
- stage3_ms_ssim_mean: 0.7348250424138446
- stage4_ms_ssim_mean: 0.7177445649884749
- stage3_lpips_alex_mean: 0.5757592567850066
- stage4_lpips_alex_mean: 0.4189543044262066
- stage3_dists_mean: 0.3536208531130915
- stage4_dists_mean: 0.28531707996043604
- stage4_l1_mean: 0.059281572556136634
- stage3_l1_mean: 0.05592005345565469
- condition_l1_mean: 0.4118557400137618
- pre_control_condition_l1_mean: 0.4118557400137618
- control_condition_l1_delta_mean: 0.0
- ungated_condition_l1_mean: 0.4118557400137618
- base_condition_l1_mean: 0.5373054925838242
- condition_l1_delta_vs_base_mean: -0.12544975257006244
- condition_residual_l1_mean: 0.3504620695999567
- condition_delta_raw_l1_mean: 0.5692594791560069
- condition_residual_guard_mean_mean: 1.0
- condition_residual_guard_min_mean: 1.0
- condition_residual_guard_max_mean: 1.0
- control_grid_abs_mean_mean: 0.0
- control_hybrid_mode_index_mean: 0.0
- control_hybrid_rdo_score_mean: 0.0
- stage4_alpha_mean: 1.0
- condition_gate_mean_mean: 0.0
- post_control_condition_gate_mean_mean: 0.0
- count: 552
- stage4_psnr_win_rate: 0.019927536231884056
- stage4_ms_ssim_win_rate: 0.11413043478260869
- stage4_lpips_win_rate: 0.9945652173913043
- stage4_dists_win_rate: 0.9927536231884058
- stage4_blend_alpha: 1.0
- stage4_alpha_min: 1.0
- stage4_alpha_max: 1.0
- stage4_alpha_std: 0.0
- stage4_gate_checkpoint: 
- stage4_condition_gate_checkpoint: 
- stage4_post_control_condition_gate_checkpoint: 
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
- control_hybrid_mode_counts: {'none': 552}
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

- summary: `results/stage4_cod_lite_adapter_eval/20260629_stage4_detailcontrast_statslite_ft500_b8_full552_eval/summary.json`
- per_image: `results/stage4_cod_lite_adapter_eval/20260629_stage4_detailcontrast_statslite_ft500_b8_full552_eval/per_image_metrics.jsonl`
- reconstructions: ``
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_155607-mt28nt5e`
