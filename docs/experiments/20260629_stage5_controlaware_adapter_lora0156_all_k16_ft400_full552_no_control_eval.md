# 20260629_stage5_controlaware_adapter_lora0156_all_k16_ft400_full552_no_control_eval

Date: 2026-06-29T21:20:05

## Command

```bash
scripts/eval_stage4_cod_lite_adapter.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage5_controlaware_adapter_lora0156_all_k16_ft400_b4ga2.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl --per-image-metrics results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/per_image_metrics.jsonl --run-name 20260629_stage5_controlaware_adapter_lora0156_all_k16_ft400_full552_no_control_eval --crop-size 512 --batch-size 8 --num-workers 4 --wandb-mode offline
```

## Summary

- actual_payload_bpp_mean: 0.013999109682829483
- stage3_actual_payload_bpp_mean: 0.013999109682829483
- control_payload_bpp_mean: 0.0
- control_payload_bytes_mean: 0.0
- stage3_psnr_mean: 21.99508854271709
- stage4_psnr_mean: 21.359426759291388
- stage3_ms_ssim_mean: 0.7348250424138446
- stage4_ms_ssim_mean: 0.7166309775514663
- stage3_lpips_alex_mean: 0.5757592567850066
- stage4_lpips_alex_mean: 0.38879488541515195
- stage3_dists_mean: 0.3536208531130915
- stage4_dists_mean: 0.2716414640131204
- stage4_l1_mean: 0.05926436196972171
- stage3_l1_mean: 0.05592005345565469
- condition_l1_mean: 0.41055272724749386
- pre_control_condition_l1_mean: 0.41055272724749386
- control_condition_l1_delta_mean: 0.0
- ungated_condition_l1_mean: 0.41055272724749386
- base_condition_l1_mean: 0.5373054925838242
- condition_l1_delta_vs_base_mean: -0.1267527653363304
- condition_residual_l1_mean: 0.3407542129735584
- condition_delta_raw_l1_mean: 0.5431615206534448
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
- split_metrics: {'clic2020_test428': {'count': 428.0, 'actual_payload_bpp_mean': 0.013746386376496788, 'paper_bpp_mean': 0.013746386376496788, 'stage3_actual_payload_bpp_mean': 0.013746386376496788, 'control_payload_bpp_mean': 0.0, 'control_payload_bytes_mean': 0.0, 'stage3_psnr_mean': 22.363807546758206, 'stage4_psnr_mean': 21.72953287686143, 'stage4_psnr_delta_vs_stage3_mean': -0.6342746698967764, 'stage3_ms_ssim_mean': 0.7453523150462413, 'stage4_ms_ssim_mean': 0.7289161509268474, 'stage4_ms_ssim_delta_vs_stage3_mean': -0.016436164119394026, 'stage3_lpips_alex_mean': 0.5609682095976912, 'stage4_lpips_alex_mean': 0.3792495131579654, 'stage4_lpips_alex_delta_vs_stage3_mean': -0.18171869643972577, 'stage3_dists_mean': 0.3499138292586692, 'stage4_dists_mean': 0.268550212834483, 'stage4_dists_delta_vs_stage3_mean': -0.08136361642418621, 'condition_l1_mean': 0.406552339268622, 'pre_control_condition_l1_mean': 0.406552339268622, 'control_condition_l1_delta_mean': 0.0, 'ungated_condition_l1_mean': 0.406552339268622, 'base_condition_l1_mean': 0.5325573511491312, 'condition_l1_delta_vs_base_mean': -0.1260050118805092, 'stage4_alpha_mean': 1.0, 'condition_gate_mean_mean': 0.0, 'post_control_condition_gate_mean_mean': 0.0, 'stage4_psnr_win_rate': 0.02570093457943925, 'stage4_ms_ssim_win_rate': 0.11682242990654206, 'stage4_lpips_win_rate': 0.9929906542056075, 'stage4_dists_win_rate': 0.9929906542056075}, 'div2k_val100': {'count': 100.0, 'actual_payload_bpp_mean': 0.01505096435546875, 'paper_bpp_mean': 0.01505096435546875, 'stage3_actual_payload_bpp_mean': 0.01505096435546875, 'control_payload_bpp_mean': 0.0, 'control_payload_bytes_mean': 0.0, 'stage3_psnr_mean': 20.49562247276306, 'stage4_psnr_mean': 19.842437944412232, 'stage4_psnr_delta_vs_stage3_mean': -0.6531845283508301, 'stage3_ms_ssim_mean': 0.6927147376537323, 'stage4_ms_ssim_mean': 0.6679971492290497, 'stage4_ms_ssim_delta_vs_stage3_mean': -0.024717588424682618, 'stage3_lpips_alex_mean': 0.6244589188694953, 'stage4_lpips_alex_mean': 0.42314624920487404, 'stage4_lpips_alex_delta_vs_stage3_mean': -0.20131266966462136, 'stage3_dists_mean': 0.364796816110611, 'stage4_dists_mean': 0.28253867268562316, 'stage4_dists_delta_vs_stage3_mean': -0.08225814342498779, 'condition_l1_mean': 0.42387099444866183, 'pre_control_condition_l1_mean': 0.42387099444866183, 'control_condition_l1_delta_mean': 0.0, 'ungated_condition_l1_mean': 0.42387099444866183, 'base_condition_l1_mean': 0.5505430579185486, 'condition_l1_delta_vs_base_mean': -0.12667206346988677, 'stage4_alpha_mean': 1.0, 'condition_gate_mean_mean': 0.0, 'post_control_condition_gate_mean_mean': 0.0, 'stage4_psnr_win_rate': 0.01, 'stage4_ms_ssim_win_rate': 0.06, 'stage4_lpips_win_rate': 1.0, 'stage4_dists_win_rate': 0.99}, 'kodak24': {'count': 24.0, 'actual_payload_bpp_mean': 0.014123280843098959, 'paper_bpp_mean': 0.014123280843098959, 'stage3_actual_payload_bpp_mean': 0.014123280843098959, 'control_payload_bpp_mean': 0.0, 'control_payload_bytes_mean': 0.0, 'stage3_psnr_mean': 21.667374928792317, 'stage4_psnr_mean': 21.07998772462209, 'stage4_psnr_delta_vs_stage3_mean': -0.587387204170227, 'stage3_ms_ssim_mean': 0.72254828363657, 'stage4_ms_ssim_mean': 0.7001863370339075, 'stage4_ms_ssim_delta_vs_stage3_mean': -0.022361946602662403, 'stage3_lpips_alex_mean': 0.6366176729400953, 'stage4_lpips_alex_mean': 0.41589000821113586, 'stage4_lpips_alex_delta_vs_stage3_mean': -0.2207276647289594, 'stage3_dists_mean': 0.3731629326939583, 'stage4_dists_mean': 0.28136374056339264, 'stage4_dists_delta_vs_stage3_mean': -0.09179919213056564, 'condition_l1_mean': 0.42640019953250885, 'pre_control_condition_l1_mean': 0.42640019953250885, 'control_condition_l1_delta_mean': 0.0, 'ungated_condition_l1_mean': 0.42640019953250885, 'base_condition_l1_mean': 0.5668241592744986, 'condition_l1_delta_vs_base_mean': -0.14042395974198976, 'stage4_alpha_mean': 1.0, 'condition_gate_mean_mean': 0.0, 'post_control_condition_gate_mean_mean': 0.0, 'stage4_psnr_win_rate': 0.0, 'stage4_ms_ssim_win_rate': 0.0, 'stage4_lpips_win_rate': 1.0, 'stage4_dists_win_rate': 1.0}}
- stage4_psnr_win_rate: 0.021739130434782608
- stage4_ms_ssim_win_rate: 0.10144927536231885
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

- summary: `results/stage4_cod_lite_adapter_eval/20260629_stage5_controlaware_adapter_lora0156_all_k16_ft400_full552_no_control_eval/summary.json`
- per_image: `results/stage4_cod_lite_adapter_eval/20260629_stage5_controlaware_adapter_lora0156_all_k16_ft400_full552_no_control_eval/per_image_metrics.jsonl`
- reconstructions: ``
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_211915-6tpitwlr`
