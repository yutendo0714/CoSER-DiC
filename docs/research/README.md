# CoSER-DiC Core MVP Research Notes

This directory is intentionally centered on Core MVP-v0.

Read in this order:

0. `current_mvp_state_20260628.md`
1. `design_decisions/024_stage4_stage5_mainline_research_direction.md`
2. `design_decisions/025_stage4_no_posthoc_rgb_mainline_condition_gate.md`
3. `design_decisions/026_stage4_coser_feature_ablation_and_guarded_adapter.md`
4. `design_decisions/027_stage4_detailaware_condition_adapter.md`
5. `design_decisions/028_stage4_rate_specific_cod_lite_backbone_probe.md`
6. `design_decisions/029_stage4_dists_loss_probe_and_traincache_scale.md`
7. `design_decisions/030_stage4_fast_train_cache_and_8192_adapter_scale.md`
8. `design_decisions/031_stage4_fast8192_coser_feature_ablation.md`
9. `design_decisions/032_stage4_semantic_dropout_detail_use_probe.md`
10. `design_decisions/033_stage4_detail_control_branch_probe.md`
11. `design_decisions/034_stage4_detail_highfreq_residual_loss_probe.md`
12. `design_decisions/035_stage4_condition_gate_dists_probe.md`
13. `design_decisions/036_stage4_detail_film_condition_adapter.md`
14. `design_decisions/037_bd_rate_tooling_and_current_external_gap.md`
15. `design_decisions/038_stage5_partial_backbone_finetune_path.md`
16. `design_decisions/039_stage5_tiny_counted_condition_control_stream.md`
17. `design_decisions/040_stage5_lora_backbone_adaptation_path.md`
18. `design_decisions/041_stage5_condition_residual_rms_guard_probe.md`
19. `design_decisions/042_stage5_anchor_aware_candidate_selection.md`
20. `design_decisions/043_gpu_preflight_for_long_experiments.md`
21. `design_decisions/044_stage5_counted_affine_condition_control.md`
22. `design_decisions/045_stage5_counted_affine_dct_condition_control.md`
23. `design_decisions/046_stage5_counted_affine_grid_condition_control.md`
24. `design_decisions/047_stage5_counted_affine_basis_condition_control.md`
25. `design_decisions/048_stage5_counted_control_master_screen_selection.md`
26. `design_decisions/049_stage5_counted_hybrid_condition_control.md`
27. `design_decisions/050_stage5_counted_hybrid_basis_condition_control.md`
28. `design_decisions/051_stage5_hybrid_image_rdo_selector.md`
29. `design_decisions/052_stage5_hybrid_selector_bit_accounting.md`
30. `design_decisions/053_stage5_guarded_image_rdo_selector.md`
31. `design_decisions/054_stage5_componentwise_basis_quantization.md`
32. `design_decisions/055_stage5_component_codebook_basis_quantization.md`
33. `design_decisions/056_stage5_component_codebook_huffman_control.md`
34. `design_decisions/057_stage5_vector_codebook_basis_control.md`
35. `design_decisions/058_stage5_compact_sparse_topk_basis_payload.md`
36. `design_decisions/059_stage4_dists_adapter_and_counted_control_guardrails.md`
37. `design_decisions/060_stage4_perceptual_adapter_and_coser_feature_ablation.md`
38. `design_decisions/061_stage4_detailcontrast_and_stage5_sparse_control.md`
39. `design_decisions/062_mainline_diagnosis_and_next_policy_20260629.md`
40. `design_decisions/063_condition_stats_ablation_and_stats_loss_probe_20260629.md`
41. `design_decisions/064_stage4_detail_residual_target_perceptual_probe_20260629.md`
42. `design_decisions/065_stage4_detailtarget_long_and_stage5_rebased_control_20260629.md`
43. `design_decisions/066_stage5_topk16_gate_regularization_probe_20260629.md`
44. `design_decisions/067_stage5_lora_all_control_gate_rate_probe_20260629.md`
45. `design_decisions/068_stage5_prefix_topk_entropy_and_spatial_gate_probe_20260629.md`
46. `design_decisions/069_stage5_controlaware_adapter_training_probe_20260629.md`
47. `design_decisions/070_stage5_basefixed_fixedtarget_and_hybrid_selector_probe_20260629.md`
48. `design_decisions/071_stage5_detailcontrol_fusion_probe_20260629.md`
49. `design_decisions/072_stage5_detail_highfreq_context_probe_20260629.md`
50. `design_decisions/073_stage5_zero_centered_detail_quantizer_probe_20260630.md`
51. `design_decisions/000_external_repos_and_pretrained_policy_mvp.md`
52. `design_decisions/001_core_mvp_focus_audit.md`
53. `design_decisions/003_official_implementation_reference_policy.md`
54. `design_decisions/004_stage1_semantic_vq_reference_audit.md`
55. `design_decisions/005_stage2_static_huffman_token_prior.md`
56. `design_decisions/006_stage2_learned_token_prior_probe.md`
57. `design_decisions/007_stage3_uniform_residual_bootstrap.md`
58. `design_decisions/008_compact_v3_actual_bitstream_container.md`
59. `design_decisions/009_stage3_semantic_conditioned_residual_huffman.md`
60. `design_decisions/010_bpp_reporting_policy.md`
61. `design_decisions/011_stage3_semantic_left_context_residual_huffman.md`
62. `design_decisions/012_evaluation_protocol_alignment.md`
63. `design_decisions/013_stage3_protocol_perceptual_evaluation.md`
64. `design_decisions/014_stage3_reconstruction_distribution_metrics.md`
65. `design_decisions/015_stage3_fixed_detail_gain_sweep.md`
66. `design_decisions/016_deterministic_actual_bpp_evaluation.md`
67. `design_decisions/017_stage3_fixed_decoder_unsharp_postprocess.md`
68. `design_decisions/018_stage1_lpips_finetune_revisit.md`
69. `design_decisions/019_stage1_rateprior_b16_reconnection.md`
70. `design_decisions/020_stage4_decoder_side_refiner.md`
71. `design_decisions/021_stage4_cod_codlite_parallel_backbone_policy.md`
72. `design_decisions/022_stage4_cod_lite_adapter_bootstrap.md`
73. `design_decisions/023_stage4_semantic_latent_cod_lite_adapter.md`
74. `core_mvp_literature_update_2026-06-27.md`
75. `baseline_registry.md`
76. `baselines/external_repo_status_mvp.md`
77. `baselines/pretrained_asset_inventory_20260628.md`
78. `baselines/cod_lite_official_baseline_20260628.md`
79. `baselines/cod_lite_full552_official_plan_20260629.md`
80. `baselines/cod_one_step_full552_official_plan_20260629.md`

For day-to-day continuation, start from `current_mvp_state_20260628.md`. It
records the active Stage 1-3 anchor, the fact that the ResUNet refiner is
archived as diagnostics only, and the active CoD-Lite / CoD Stage 4 direction.
The mainline guardrails for what to optimize next are in
`design_decisions/024_stage4_stage5_mainline_research_direction.md`. The
specific no-posthoc-RGB rule for Stage 4/5 is in
`design_decisions/025_stage4_no_posthoc_rgb_mainline_condition_gate.md`. The
current CoSER feature-use ablation and guarded adapter training update is in
`design_decisions/026_stage4_coser_feature_ablation_and_guarded_adapter.md`.
The current best no-extra-bit raw CoD-Lite adapter baseline is the
detail-aware condition adapter in
`design_decisions/027_stage4_detailaware_condition_adapter.md`. The current
rate-specific CoD-Lite backbone comparison is in
`design_decisions/028_stage4_rate_specific_cod_lite_backbone_probe.md`. The
first DISTS-loss continuation probe and the decision to prioritize larger
clean train-cache scaling are in
`design_decisions/029_stage4_dists_loss_probe_and_traincache_scale.md`. The
fast Stage 3 training cache and the 8192-image Stage 4 adapter-scale result are
in `design_decisions/030_stage4_fast_train_cache_and_8192_adapter_scale.md`.
The fast8192 semantic/detail feature-use ablation and the current diagnosis
that detail context is still under-used are in
`design_decisions/031_stage4_fast8192_coser_feature_ablation.md`.
The semantic-latent dropout probe is in
`design_decisions/032_stage4_semantic_dropout_detail_use_probe.md`; it improves
LPIPS/DISTS slightly but does not materially increase detail-context use, so
the next mainline move is a stronger detail-control condition branch.
The first direct detail-control branch probe is in
`design_decisions/033_stage4_detail_control_branch_probe.md`; it improves
condition L1 but still does not materially increase detail-context use, so it
is infrastructure rather than a promoted Stage 4/5 result.
The detail high-frequency residual loss probe is in
`design_decisions/034_stage4_detail_highfreq_residual_loss_probe.md`; it
increases detail ablation sensitivity but hurts LPIPS/DISTS in the first
limit64 probes, so it remains diagnostic infrastructure rather than a promoted
method.
The DISTS-aware condition-space gate probe is in
`design_decisions/035_stage4_condition_gate_dists_probe.md`; it slightly
recovers fidelity versus the raw adapter but worsens LPIPS/DISTS and collapses
near the raw branch, so the next mainline move is to strengthen condition
adapter information flow instead of only gating an imperfect residual.
The detail-FiLM condition adapter is in
`design_decisions/036_stage4_detail_film_condition_adapter.md`; it gives a
small full552 improvement over the semantic-dropout detail-control raw adapter
at unchanged actual payload bpp, so it is current mainline infrastructure but
not a Stage 5 or external-baseline result.
The current detail-contrast Stage 4 anchor, no-extra-bit balanced condition
gate operating point, and sparse Stage 5 counted-control anchors are in
`design_decisions/061_stage4_detailcontrast_and_stage5_sparse_control.md`.
The current high-level diagnosis and next research policy, including the
decision to prioritize condition imitation, condition statistics/spectrum
matching, diffusion-friendly detail control, actual-payload curves, and matched
official baselines over fixed alpha/postprocessing tricks, is in
`design_decisions/062_mainline_diagnosis_and_next_policy_20260629.md`.
The full552 condition-statistics feature ablation and stats/spectrum-loss
finetune probe are in
`design_decisions/063_condition_stats_ablation_and_stats_loss_probe_20260629.md`;
it shows semantic latent is used, detail context is still under-used, and
stats-only tuning should remain auxiliary rather than the main perceptual
improvement path.
The detail-residual target and perceptual-balanced probe is in
`design_decisions/064_stage4_detail_residual_target_perceptual_probe_20260629.md`;
it adds `--detail-residual-target-weight`, promotes
`20260629_stage4_detailtarget_perceptual_ft700_b8` as the current no-extra-bit
Stage 4 candidate, and documents the small but real full552 improvement plus
detail-use ablations.
The long detail-target continuation and rebased Stage 5 counted-control update
is in
`design_decisions/065_stage4_detailtarget_long_and_stage5_rebased_control_20260629.md`;
it promotes `20260629_stage4_detailtarget_perceptual_long_ft1800_b8` as the
current no-extra-bit Stage 4 anchor, records the rebased sparse top-k32 control
path, promotes a balanced post-control gate anchor, and logs GenCodec-style
patch-FID/KID showing large internal distribution gains over Stage 3 while
still not closing the official CoD-Lite gap.
The top-k16 gate regularization probe is in
`design_decisions/066_stage5_topk16_gate_regularization_probe_20260629.md`;
it adds optional gate deviation/TV regularization and shows that smoothing the
current deterministic condition gate trades perceptual gains for fidelity, so
the next mainline push should improve adapter/control representation rather
than keep tuning gate-only sweeps.
The denoiser-all LoRA, k16 counted-control, learned global condition gate, and
0.0078 rate-checkpoint rejection are in
`design_decisions/067_stage5_lora_all_control_gate_rate_probe_20260629.md`;
it promotes the 0.0156 denoiser-all LoRA branch as the active Stage 5
continuation, identifies k16 sparse control as the most efficient current
counted-control point, keeps the learned global gate as a balanced condition
space operating point, and rejects the 0.0078 backbone branch.
The prefix-top-k entropy and spatial condition-gate probe is in
`design_decisions/068_stage5_prefix_topk_entropy_and_spatial_gate_probe_20260629.md`;
it adds a valid entropy-coded `prefix_topk` control path and tests a spatial
condition gate, but does not promote either as the main result because the
improvements are small and do not close the external CoD-Lite gap. The next
mainline push should train a stronger control-aware adapter/detail-control
path rather than continue gate-only tuning.
The first control-aware adapter training probe is in
`design_decisions/069_stage5_controlaware_adapter_training_probe_20260629.md`;
it adds `--train-counted-control-mode condition_residual_affine_basis`, proves
that the counted k16 control path can be included during adapter training, but
does not promote the FT400 checkpoint because DISTS and patch-FID/KID do not
beat the stronger existing anchors.
The base-fixed control, corrected fixed-target control-aware retraining, and
hybrid selector follow-up is in
`design_decisions/070_stage5_basefixed_fixedtarget_and_hybrid_selector_probe_20260629.md`;
it keeps the fixed-target training implementation because it improves
condition/LPIPS/fidelity diagnostics, but does not promote base-fixed,
DISTS-reweighted, or selector-only branches because none produces a clear
full552 Pareto improvement over the active k16 counted-control anchor.
The direct detail-control fusion and high-frequency detail-context probes are
in `design_decisions/071_stage5_detailcontrol_fusion_probe_20260629.md` and
`design_decisions/072_stage5_detail_highfreq_context_probe_20260629.md`; they
add useful no-extra-bit adapter hooks but do not close the curve gap, so the
next mainline move is to improve the decoded detail payload itself.
The zero-centered detail quantizer probe is in
`design_decisions/073_stage5_zero_centered_detail_quantizer_probe_20260630.md`;
it fixes the low-bit residual-grid zero-code mismatch, makes d16/b2 detail
coding substantially stronger, and keeps the quantizer infrastructure, but it
does not promote the d16/b2 checkpoint as a Stage 5 win because it spends more
actual bpp than the d32 anchor without dominating the curve.
BD-rate tooling and the current external gap are in
`design_decisions/037_bd_rate_tooling_and_current_external_gap.md`; current
Kodak LPIPS/DISTS do not overlap the official CoD-Lite Kodak512 curve, so
Stage 5 work must target a large perceptual jump before BD-rate can even be
computed.
The Stage 5 partial-backbone finetune path is in
`design_decisions/038_stage5_partial_backbone_finetune_path.md`; it adds a
low-LR decoder adaptation route that keeps CoSER semantic/detail streams as the
only transmitted payload and treats adapted backbone tensors as fixed model
state, not per-image bits.
The tiny counted condition-control stream path is in
`design_decisions/039_stage5_tiny_counted_condition_control_stream.md`; it
adds explicitly entropy-coded per-image grid/DCT/basis control payloads and
makes `actual_payload_bpp` include those bytes.
The Stage 5 LoRA backbone adaptation path is in
`design_decisions/040_stage5_lora_backbone_adaptation_path.md`; it adds a
low-rank fixed-decoder-state adaptation route for official CoD-Lite modules
while keeping CoSER semantic/detail payload as the transmitted codec stream.
The Stage 5 condition residual RMS guard probe is in
`design_decisions/041_stage5_condition_residual_rms_guard_probe.md`; it adds a
deterministic condition-space stability sweep, not final RGB postprocessing,
with no additional actual payload bpp.
The Stage 5 anchor-aware candidate selection update is in
`design_decisions/042_stage5_anchor_aware_candidate_selection.md`; it makes
limit64 and full552 promotion compare candidates against the current
detail-FiLM anchor, not only against Stage 3.
The GPU preflight policy for long-running generated shell plans is in
`design_decisions/043_gpu_preflight_for_long_experiments.md`; it makes Stage 5
and official baseline plans fail fast when CUDA/NVML is unavailable.
The counted affine condition-control Stage 5 path is in
`design_decisions/044_stage5_counted_affine_condition_control.md`; it adds a
tiny actual-payload stream that transmits group-wise gain/bias corrections for
the CoD-Lite condition residual. The current GPU-restart entrypoint is the
curve-oriented six-candidate shell plan at
`results/stage5_counted_control/20260629_detailfilm_affine_curve_limit64_sweep_plan.sh`.
The affine+DCT extension is in
`design_decisions/045_stage5_counted_affine_dct_condition_control.md`; it first
corrects condition scale/bias and then sends a tiny grouped DCT residual for the
remaining condition error. Its current GPU-restart entrypoint is
`results/stage5_counted_control/20260629_detailfilm_affinedct_limit64_sweep_plan.sh`.
The affine+grid extension is in
`design_decisions/046_stage5_counted_affine_grid_condition_control.md`; it uses
a more flexible grouped grid residual after the affine correction. Its current
GPU-restart entrypoint is
`results/stage5_counted_control/20260629_detailfilm_affinegrid_limit64_sweep_plan.sh`.
The affine+basis extension is in
`design_decisions/047_stage5_counted_affine_basis_condition_control.md`; it fits
a post-affine residual basis from the non-eval train cache and then sweeps
counted affine+basis controls. Its GPU-restart prepare entrypoint is
`results/stage5_counted_control/20260629_detailfilm_postaffine_basis_prepare_plan.sh`.
The counted-control master screen/select policy is in
`design_decisions/048_stage5_counted_control_master_screen_selection.md`.
The counted hybrid condition-control path is in
`design_decisions/049_stage5_counted_hybrid_condition_control.md`; it transmits
a per-image mode selector plus the selected none/affine/affine+DCT/affine+grid
payload, all counted in actual payload bpp.
The counted hybrid-basis extension is in
`design_decisions/050_stage5_counted_hybrid_basis_condition_control.md`; it
adds affine+basis as another image-selectable payload family after post-affine
basis fitting.
The hybrid image-RDO selector is in
`design_decisions/051_stage5_hybrid_image_rdo_selector.md`; it lets the encoder
choose a counted hybrid control mode by decoded-image proxies such as image L1,
LPIPS, or DISTS rather than condition L1 alone.
Hybrid selector bit accounting is in
`design_decisions/052_stage5_hybrid_selector_bit_accounting.md`; it separates
selector bits from payload byte rounding and keeps Huffman/top-k accounting
conservative. The guarded image-RDO selector is in
`design_decisions/053_stage5_guarded_image_rdo_selector.md`; it adds an
encoder-side fidelity term to LPIPS/DISTS RDO so perceptual control choices do
not freely damage structure.
Component-wise basis quantization is in
`design_decisions/054_stage5_componentwise_basis_quantization.md`; it lets
fixed-bit prefix/top-k basis controls use decoder-side per-component p95/p99
ranges to improve condition-control quantization at unchanged payload bytes.
Component codebook basis quantization is in
`design_decisions/055_stage5_component_codebook_basis_quantization.md`; it
fits fixed decoder-side per-component scalar codebooks on non-eval train-cache
coefficients, while per-image payload remains only fixed-bit coefficient level
indices.
Component codebook Huffman control is in
`design_decisions/056_stage5_component_codebook_huffman_control.md`; it adds
static Huffman coding for those codebook level symbols so the same decoded
condition-control coefficients can be tested at lower actual payload bpp.
Vector-codebook basis control is in
`design_decisions/057_stage5_vector_codebook_basis_control.md`; it transmits a
single fixed/Huffman-coded vector index for a coefficient prefix, giving very
low-control-bpp curve points and hybrid candidates.
Compact sparse top-k basis payload accounting is in
`design_decisions/058_stage5_compact_sparse_topk_basis_payload.md`; it packs
fixed-bit top-k index/value codes into one payload stream so low-rate
actual_payload_bpp is counted exactly instead of being inflated by separate
byte rounding.
The DISTS-loss Stage 4 adapter continuation and counted-control guardrail
update is in
`design_decisions/059_stage4_dists_adapter_and_counted_control_guardrails.md`;
it promotes a small no-extra-bit adapter improvement, records the current
fidelity/perception tradeoff for sparse counted control, and fixes bpp/limit
tooling traps.
The detail-contrast/high-frequency Stage 4 adapter and refreshed sparse
top-k32 Stage 5 counted-control anchor are in
`design_decisions/061_stage4_detailcontrast_and_stage5_sparse_control.md`;
this is the current internal mainline anchor and includes GenCodec-style
patch-FID/KID for the sc1.25 perceptual control point.
The current preferred Stage 5 counted-control GPU-restart entrypoint is the
master screen/select shell at
`results/stage5_counted_control/20260629_stage5_mainline_screen_select_plan.sh`;
it runs affine, affine+DCT, affine+grid, hybrid, image-RDO hybrid,
guarded image-RDO hybrid, post-affine basis fit, generated affine+basis /
hybrid-basis / DISTS-RDO hybrid-basis screening, cross-family anchor-aware
selection, and full552 promotion-plan generation. Inspect
`results/stage5_selection/20260629_stage5_mainline_limit64_all_control_selection.json`
before running the generated full552 promotion shell.
Official CoD-Lite Kodak512 baseline results and the pending full552 baseline
reproduction plan live in `baselines/cod_lite_official_baseline_20260628.md`
and `baselines/cod_lite_full552_official_plan_20260629.md`. The parallel CoD
one-step baseline plan is in
`baselines/cod_one_step_full552_official_plan_20260629.md`. Use these before
claiming any Stage 5 external-baseline win.

Broad literature notes that are not part of the active Core MVP execution path
live in `archive/`.
