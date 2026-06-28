# CoSER-DiC Core MVP Research Notes

This directory is intentionally centered on Core MVP-v0.

Read in this order:

0. `current_mvp_state_20260628.md`
1. `design_decisions/024_stage4_stage5_mainline_research_direction.md`
2. `design_decisions/000_external_repos_and_pretrained_policy_mvp.md`
3. `design_decisions/001_core_mvp_focus_audit.md`
4. `design_decisions/003_official_implementation_reference_policy.md`
5. `design_decisions/004_stage1_semantic_vq_reference_audit.md`
6. `design_decisions/005_stage2_static_huffman_token_prior.md`
7. `design_decisions/006_stage2_learned_token_prior_probe.md`
8. `design_decisions/007_stage3_uniform_residual_bootstrap.md`
9. `design_decisions/008_compact_v3_actual_bitstream_container.md`
10. `design_decisions/009_stage3_semantic_conditioned_residual_huffman.md`
11. `design_decisions/010_bpp_reporting_policy.md`
12. `design_decisions/011_stage3_semantic_left_context_residual_huffman.md`
13. `design_decisions/012_evaluation_protocol_alignment.md`
14. `design_decisions/013_stage3_protocol_perceptual_evaluation.md`
15. `design_decisions/014_stage3_reconstruction_distribution_metrics.md`
16. `design_decisions/015_stage3_fixed_detail_gain_sweep.md`
17. `design_decisions/016_deterministic_actual_bpp_evaluation.md`
18. `design_decisions/017_stage3_fixed_decoder_unsharp_postprocess.md`
19. `design_decisions/018_stage1_lpips_finetune_revisit.md`
20. `design_decisions/019_stage1_rateprior_b16_reconnection.md`
21. `design_decisions/020_stage4_decoder_side_refiner.md`
22. `design_decisions/021_stage4_cod_codlite_parallel_backbone_policy.md`
23. `design_decisions/022_stage4_cod_lite_adapter_bootstrap.md`
24. `design_decisions/023_stage4_semantic_latent_cod_lite_adapter.md`
25. `core_mvp_literature_update_2026-06-27.md`
26. `baseline_registry.md`
27. `baselines/external_repo_status_mvp.md`
28. `baselines/pretrained_asset_inventory_20260628.md`

For day-to-day continuation, start from `current_mvp_state_20260628.md`. It
records the active Stage 1-3 anchor, the fact that the ResUNet refiner is
archived as diagnostics only, and the active CoD-Lite / CoD Stage 4 direction.
The mainline guardrails for what to optimize next are in
`design_decisions/024_stage4_stage5_mainline_research_direction.md`.

Broad literature notes that are not part of the active Core MVP execution path
live in `archive/`.
