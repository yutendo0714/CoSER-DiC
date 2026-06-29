# 20260629 Stage 5 Counted Control Basis CPU Scaffold

Date: 2026-06-29

## Status

GPU is not visible:

```text
nvidia-smi: Failed to initialize NVML: Unknown Error
torch.cuda.is_available: False
torch.version.cuda: 12.4
```

No GPU training or evaluation was launched.

## Mainline Purpose

This is Stage 5 mainline infrastructure, not RGB output postprocessing.

The path is:

```text
decoded CoSER semantic/detail payload
  -> CoSER condition adapter
  -> CoD-Lite condition estimate
  + entropy-coded per-image condition-control coefficients
  -> CoD-Lite decoder backbone
```

The control stream is image-specific, so it is counted in
`actual_payload_bpp / paper_bpp`.

## Implemented

Added fixed-basis condition control:

```text
src/coserdic/entropy/control_grid.py
scripts/eval_stage4_cod_lite_adapter.py
scripts/fit_stage5_condition_control_basis.py
scripts/inspect_stage5_control_basis.py
scripts/sweep_stage5_counted_control.py
scripts/plan_stage5_control_curve.py
scripts/select_stage5_control_candidates.py
scripts/promote_stage5_control_candidates.py
```

Basis mode:

```text
--counted-control-mode condition_residual_basis
--control-basis <fixed decoder-side basis checkpoint>
--control-basis-components <transmitted coefficient count>
```

The fitted basis/mean are fixed decoder-side model state. Per image, only the
quantized basis coefficients are transmitted and counted.

The sweep script can now read `recommended_basis_settings.json` from the inspect
step through `--settings-json`, so p95/p99 ranges and Huffman keys do not need
to be copied into sweep commands by hand.

Sparse top-k basis selection was added as a stronger same-family Stage 5
control option:

```text
selection=prefix:
  transmit the first K fixed basis coefficients

selection=topk:
  transmit image-specific top-K indices from a fixed candidate prefix C, plus
  their quantized values
```

For `topk`, both the selected index stream and value stream are counted in
`actual_payload_bpp`. This is still a CoSER condition-control bitstream, not an
RGB postprocess or hidden decoder-side side channel.

Sparse top-k can now use either fixed bits or static Huffman priors:

```text
fixed_bits:
  fixed-bit top-K index stream + fixed-bit value stream

huffman:
  position-conditioned static Huffman top-K index stream
  + position-conditioned static Huffman value stream
```

The sparse top-k Huffman prior is fitted from non-eval train-cache top-k
selections and stored as fixed decoder-side state. Per-image index/value
payload bytes are still counted exactly.

The basis fit stores coefficient distribution statistics:

```text
coefficient_abs_mean/std/max
coefficient_abs_quantiles: p50 / p75 / p90 / p95 / p99
coefficient_component_abs_p95 / p99
control_huffman_priors
  prefix_mean_payload_bytes / min / max for each coefficient prefix
  quantizer / mu for each prior
  quantization_mae / quantization_rmse / clipped_fraction on train cache
sparse_topk_control_priors
  top-k index/value Huffman payload statistics
basis_reconstruction_stats
  prefix/top-k retained condition-residual coefficient energy
  residual energy fraction
  retained/residual L2 diagnostics
```

Use these to choose `--control-range` before test-set sweeps. The same fit now
also stores static position-conditioned Huffman priors such as `p95_b4` and
`p99_b4`; these are fixed decoder-side entropy models for the quantized basis
coefficients.

The scalar quantizer can be either `uniform` or deterministic `mu_law`.
Mu-law spends more levels near zero and stores only fixed decoder-side
parameters. Mu-law Huffman prior keys use labels such as `p95_b4_mulaw16`.

Evaluation supports:

```text
--control-codec fixed_bits
--control-codec huffman
--control-huffman-key p95_b4
--control-quantizer uniform
--control-quantizer mu_law
--control-mu 16
```

In Huffman mode, payload length is image-dependent. Eval rows count the exact
per-image Huffman bytes in `control_payload_bpp`. Inspect/sweep planning uses
the train-cache mean payload bytes for the requested coefficient prefix.

## CPU Verification

Passed:

```bash
.venv/bin/python -m py_compile \
  src/coserdic/entropy/control_grid.py \
  src/coserdic/entropy/__init__.py \
  scripts/eval_stage4_cod_lite_adapter.py \
  scripts/sweep_stage5_counted_control.py \
  scripts/fit_stage5_condition_control_basis.py
```

Passed:

```bash
PYTHONPATH=/workspace/CoSER-DiC .venv/bin/pytest \
  tests/test_control_grid.py \
  tests/test_sweep_stage5_counted_control.py \
  tests/test_fit_stage5_condition_control_basis.py \
  tests/test_bitstream.py \
  tests/test_gencodec_backbone.py \
  tests/test_compute_bd_rate.py \
  tests/test_collect_bd_curve_points.py \
  tests/test_summarize_per_image_metrics.py \
  tests/test_image_distribution_metrics.py -q
```

Result:

```text
57 passed
```

`git diff --check` also passed.

## Dry-Run Payload Checks

Dry-run command generation confirmed:

```text
basis g16 grid8 coeffs8 bits4:
  4 bytes, 0.0001220703125 bpp at 512x512

DCT g16 grid8 coeffs4 bits4:
  32 bytes, 0.0009765625 bpp at 512x512

grid g8 grid4 bits4:
  64 bytes, 0.001953125 bpp at 512x512
```

Later CPU continuation added mu-law scalar quantization:

```text
uniform + fixed_bits
uniform + huffman
mu_law + fixed_bits
mu_law + huffman
```

Fake-basis dry-run confirmed that `inspect_stage5_control_basis.py` emits all
four families and that `sweep_stage5_counted_control.py` forwards:

```text
--control-codec huffman
--control-quantizer mu_law
--control-mu 16
--control-huffman-key p95_b4_mulaw16
```

Verification after this extension:

```text
related tests: 19 passed
broader regression set: 63 passed
git diff --check: passed
```

## Selector-Bit Accounting Continuation

Added explicit hybrid selector-bit accounting:

```text
--control-hybrid-selector-bytes
--control-hybrid-selector-bits
```

Default behavior remains conservative byte accounting:

```text
selector_bits=0:
  selected payload bytes = control payload bytes + selector_bytes
```

When `selector_bits > 0` and exact fixed-bit payload size is known:

```text
selected payload bytes = ceil((payload_bits + selector_bits) / 8)
```

with a guard that never drops below the already simulated payload byte count.
Variable-length Huffman/top-k payloads still fall back to byte-granular
selector accounting.

Generated an audit dry-run plan:

```text
results/stage5_counted_control/20260629_detailfilm_hybrid_selectorbits_limit64_sweep_plan.json
results/stage5_counted_control/20260629_detailfilm_hybrid_selectorbits_limit64_sweep_plan.sh
```

Current main 4-bit hybrid settings remain byte-aligned:

```text
selector_bytes=1, selector_bits=0 -> control_bytes=25
selector_bytes=1, selector_bits=2 -> control_bytes=25
```

So this does not create a cosmetic bpp win. It makes the accounting explicit
and prepares future non-byte-aligned control streams.

Verification:

```text
targeted selector/control tests: 32 passed
git diff --check: passed
```

## Guarded Image-RDO Continuation

Added fidelity-guarded image-RDO selection for hybrid condition control:

```text
--control-hybrid-selection-objective dists|lpips_alex
--control-hybrid-fidelity-lambda <float>
--control-hybrid-fidelity-metric image_l1|image_mse
```

The encoder-side score is:

```text
image objective
+ fidelity_lambda * image fidelity guard
+ rate_lambda * selected control bpp
```

This is still a counted control-stream decision. The decoder receives only the
selected mode selector and selected control payload.

Generated dry-run plan:

```text
results/stage5_counted_control/20260629_detailfilm_hybrid_imagerdo_guarded_limit64_sweep_plan.json
results/stage5_counted_control/20260629_detailfilm_hybrid_imagerdo_guarded_limit64_sweep_plan.sh
```

The guarded plan is included in:

```text
results/stage5_counted_control/20260629_stage5_mainline_screen_select_plan.sh
```

Verification after guarded-RDO implementation:

```text
related selector/control tests: 34 passed
```

## Component-Wise Basis Quantization Continuation

Added component-wise fixed-bit quantizers for prefix basis controls:

```text
basis_range_mode=global
basis_range_mode=component_p95
basis_range_mode=component_p99
```

The component ranges are fixed decoder-side state from the fitted non-eval
train-cache basis checkpoint:

```text
coefficient_component_abs_p95
coefficient_component_abs_p99
```

No extra image-specific bits are transmitted. For prefix fixed-bit basis
controls:

```text
payload_bytes = ceil(components * bits / 8)
```

is unchanged versus global range. The intended gain is better condition-control
coefficient reconstruction at the same actual payload bpp.

The post-affine basis prepare plan now emits global/component range candidates:

```text
--range-mode global
--range-mode component_p95
--range-mode component_p99
```

Scope is intentionally conservative:

```text
enabled:
  fixed_bits + prefix basis
  fixed_bits + sparse top-k basis

not yet enabled:
  Huffman basis
```

because Huffman variants require component/index-conditioned entropy priors
before they can be counted safely.

For sparse top-k fixed-bit basis, the transmitted index stream lets the decoder
select the corresponding fixed component ranges for the value stream. This adds
no extra image-specific side information.

Verification after component-wise quantizer implementation:

```text
related control/inspect/sweep tests: 54 passed
git diff --check: passed
```

Promotion-command continuation verification:

```text
py_compile:
  scripts/promote_stage5_control_candidates.py
  scripts/select_stage5_control_candidates.py
  scripts/sweep_stage5_counted_control.py

candidate-selection / promotion tests:
  7 passed

broader Stage5 / evaluation regression:
  84 passed

dry-run:
  promote_stage5_control_candidates.py rewrites limit64 to full552 commands
  collect_bd_curve_points.py can infer promoted full552 summary paths from a
  promotion JSON and collect actual_payload_bpp curve points
  collect_bd_curve_points.py can also merge separately computed patch-FID/KID
  JSON metrics into a labeled curve point
  compute_bd_rate.py emits dataset / bpp-policy mismatch warnings and can fail
  strict promotion checks with --fail-on-dataset-mismatch
  LoRA backbone adaptation helpers were added for Stage 5 no-extra-payload
  CoD-Lite decoder adaptation
  inspect_cod_lite_backbone_params.py now works on CPU and reports LoRA
  candidate modules / rank-specific parameter counts
  plan_stage5_lora_targets.py generated bpp0.0312 rank-4 target presets from
  the CPU inspection JSON
  sweep_stage5_lora_backbone.py generated train + limit64 eval commands for
  denoiser_tail and dec_net LoRA probes
  sweep_stage5_counted_control.py reads inspect-style settings JSON directly
  inspect -> sweep sparse top-k basis settings generate eval commands with
  --control-basis-selection topk and counted candidate index payload
  inspect -> sweep sparse top-k Huffman settings generate eval commands with
  --control-codec huffman and --control-huffman-key topk_c*_k*_*
  inspect -> sweep preserves basis_reconstruction_stats fields such as
  basis_retained_energy_fraction_mean in the sweep plan
  sweep_stage5_counted_control.py can filter settings before limit64 by
  retained/residual basis energy, quantization_rmse, clipped_fraction, and
  retained_per_bpp ranking
  plan_stage5_control_curve.py selects multi-rate settings from inspected
  control-bpp bands for BD-rate curve preparation
  select_stage5_control_candidates.py applies retained/residual basis energy
  guards and excludes guard-failing rows from recommended
  collect_bd_curve_points.py now accepts --bpp-policy so official CoD-Lite
  curves are not mislabeled as CoSER payload curves
  eval_cod_lite_official_baseline.py now accepts --official-module,
  --official-repo, and --codec-name, so the same actual-payload evaluator can
  run official CoD-Lite and CoD one-step CLIs
  plan_cod_lite_official_baselines.py generated protocol-split official
  CoD-Lite full552 baseline commands and curve-collection commands
  the same planner now supports --codec cod_one_step and generated official
  CoD one-step full552 baseline commands

git diff --check:
  passed

GPU:
  nvidia-smi still fails with NVML unknown error
  torch.cuda.is_available remains false
```

GPU preflight continuation:

```text
scripts/check_gpu_ready.py was added so generated long-running GPU shell plans
fail before training/evaluation when CUDA/NVML is unavailable.

current environment:
  check_gpu_ready.py exits nonzero
  torch.cuda.is_available() is false
  torch.cuda.device_count() is 0
  nvidia-smi reports Failed to initialize NVML

regenerated shell plans now include:
  .venv/bin/python scripts/check_gpu_ready.py --min-devices 1

affected current plans:
  results/stage5_condition_guard/20260629_detailfilm_guard_screen64_plan.sh
  results/stage5_condition_guard/20260629_detailfilm_guard_full552_plan.sh
  results/stage5_lora/20260629_detailfilm_lora_guarded_denoiser_tail_decnet_limit64_plan.sh
  results/baselines/cod_lite_official/20260629_cod512_full552_plan.sh
  results/baselines/cod_one_step_official/20260629_cod512_full552_plan.sh

direct sweep execution:
  sweep_stage5_condition_residual_guard.py and sweep_stage5_counted_control.py
  also run the preflight once before launching eval commands unless
  --skip-gpu-preflight is set
```

Counted affine condition-control continuation:

```text
scripts/eval_stage4_cod_lite_adapter.py now supports:
  --counted-control-mode condition_residual_affine

This transmits per-image quantized affine gain/bias controls for:
  target_cond - base_cond ~= gain * (pred_cond - base_cond) + bias

It is a condition-space information-flow change, not RGB postprocessing.
The gain/bias payload bytes are added to actual_payload_bpp.
```

Generated affine limit64 sweep:

```text
results/stage5_counted_control/20260629_detailfilm_affine_limit64_sweep_plan.json
results/stage5_counted_control/20260629_detailfilm_affine_limit64_sweep_plan.sh
```

Planned control bpp range:

```text
0.000244140625 to 0.001953125
```

CPU verification:

```text
tests/test_eval_stage4_cod_lite_adapter_control.py
tests/test_sweep_stage5_counted_control.py
tests/test_select_stage5_control_candidates.py

24 passed
```

The sweep/curve-planning pipeline now round-trips generated dict-style
`setting` rows as reusable string settings. This matters for Stage 5 because a
generated limit64 plan can be filtered into bpp bands and replanned cleanly
without copying stale `command`, `summary`, or `run_name` metadata.

Added:

```text
scripts/sweep_stage5_counted_control.py:
  setting dict/string normalization
  --include-mode

scripts/plan_stage5_control_curve.py:
  --include-mode
```

Generated affine curve-oriented candidate settings:

```text
results/stage5_counted_control/20260629_detailfilm_affine_limit64_curve_settings.json
```

Bands:

```text
0.0000 <= control_bpp < 0.0005: 2 settings
0.0005 <= control_bpp < 0.0011: 2 settings
0.0011 <= control_bpp < 0.0021: 2 settings
```

Generated clean executable plan:

```text
results/stage5_counted_control/20260629_detailfilm_affine_curve_limit64_sweep_plan.json
results/stage5_counted_control/20260629_detailfilm_affine_curve_limit64_sweep_plan.sh
```

The shell plan includes GPU preflight:

```text
.venv/bin/python scripts/check_gpu_ready.py --min-devices 1
```

Pending-selection dry-run confirmed that the six expected summaries are
currently missing and the matching detail-FiLM limit64 anchor loads:

```text
input_count: 6
loaded_count: 0
anchor count: 64
missing_summaries: 6
```

Updated CPU verification:

```text
tests/test_sweep_stage5_counted_control.py
tests/test_plan_stage5_control_curve.py

17 passed
```

Counted affine+DCT continuation:

```text
scripts/eval_stage4_cod_lite_adapter.py now supports:
  --counted-control-mode condition_residual_affine_dct
```

Purpose:

```text
first use a tiny affine gain/bias stream to fix condition residual scale/bias,
then DCT-code the remaining grouped condition error
```

This is a condition-space control stream. It is not final RGB blending, and all
affine + DCT bytes are added to `actual_payload_bpp`.

Generated pending limit64 plan:

```text
results/stage5_counted_control/20260629_detailfilm_affinedct_limit64_sweep_plan.json
results/stage5_counted_control/20260629_detailfilm_affinedct_limit64_sweep_plan.sh
```

Planned payload:

```text
8 settings
control_bpp: 0.00048828125 to 0.00146484375
```

Pending-selection dry-run confirmed:

```text
input_count: 8
loaded_count: 0
anchor count: 64
missing_summaries: 8
```

CPU verification after affine+DCT extension:

```text
tests/test_eval_stage4_cod_lite_adapter_control.py
tests/test_sweep_stage5_counted_control.py
tests/test_select_stage5_control_candidates.py

28 passed
```

Counted affine+grid continuation:

```text
scripts/eval_stage4_cod_lite_adapter.py now supports:
  --counted-control-mode condition_residual_affine_grid
```

Purpose:

```text
first use a tiny affine gain/bias stream to fix condition residual scale/bias,
then grid-code the remaining grouped condition error
```

Compared with affine+DCT:

```text
affine+DCT:
  lower-rate, low-frequency residual correction

affine+grid:
  higher-flexibility grouped spatial residual correction
```

Generated pending limit64 plan:

```text
results/stage5_counted_control/20260629_detailfilm_affinegrid_limit64_sweep_plan.json
results/stage5_counted_control/20260629_detailfilm_affinegrid_limit64_sweep_plan.sh
```

Planned payload:

```text
8 settings
control_bpp: 0.0003662109375 to 0.002197265625
```

Pending-selection dry-run confirmed:

```text
input_count: 8
loaded_count: 0
anchor count: 64
missing_summaries: 8
```

CPU verification after affine+grid extension:

```text
tests/test_eval_stage4_cod_lite_adapter_control.py
tests/test_sweep_stage5_counted_control.py

18 passed
```

Counted affine+basis continuation:

```text
scripts/eval_stage4_cod_lite_adapter.py now supports:
  --counted-control-mode condition_residual_affine_basis

scripts/fit_stage5_condition_control_basis.py now supports:
  --pre-basis-affine true

scripts/sweep_stage5_counted_control.py now supports:
  mode=affine_basis
  --override-mode affine_basis
```

Purpose:

```text
first use a tiny affine gain/bias stream to fix condition residual scale/bias,
then use a train-cache-fitted basis to code the remaining condition error
```

Why this matters:

```text
raw basis control may waste components on affine-like scale/bias mismatch
post-affine basis fitting should spend components on the residual structure
that affine cannot represent
```

Generated prepare plan:

```text
results/stage5_counted_control/20260629_detailfilm_postaffine_basis_prepare_plan.sh
```

The prepare plan will:

```text
fit post-affine basis on the 8192-image non-eval train cache
inspect recommended basis settings
generate an affine_basis limit64 sweep plan
```

Expected outputs after GPU execution:

```text
outputs/stage5_control_basis/20260629_detailfilm_postaffine_basis_g16s8_k64_train8192/control_basis.pt
outputs/stage5_control_basis/20260629_detailfilm_postaffine_basis_g16s8_k64_train8192/recommended_basis_settings.json
results/stage5_counted_control/20260629_detailfilm_affinebasis_limit64_sweep_plan.json
results/stage5_counted_control/20260629_detailfilm_affinebasis_limit64_sweep_plan.sh
```

CPU verification after affine+basis extension:

```text
tests/test_eval_stage4_cod_lite_adapter_control.py
tests/test_sweep_stage5_counted_control.py
tests/test_fit_stage5_condition_control_basis.py

27 passed
```

Mainline counted-control screen/select master plan:

```text
docs/research/design_decisions/048_stage5_counted_control_master_screen_selection.md
results/stage5_counted_control/20260629_stage5_mainline_screen_select_plan.sh
```

Purpose:

```text
run the affine curve screen
run affine+DCT and affine+grid screens
fit post-affine basis on non-eval train cache
generate and run the affine+basis screen
select all executed candidates against the detail-FiLM limit64 anchor
generate full552 promotion commands for only the selected candidates
```

This is the preferred GPU-restart entrypoint for the current counted-control
family. It keeps all candidates in the same anchor-aware ranking pass instead
of manually promoting one family at a time.

The plan starts with:

```bash
.venv/bin/python scripts/check_gpu_ready.py --min-devices 1
```

so it fails fast while CUDA/NVML is unavailable.

Expected generated selection/promotion artifacts:

```text
results/stage5_selection/20260629_stage5_mainline_limit64_all_control_selection.json
results/stage5_selection/20260629_stage5_mainline_full552_promotion_plan.json
results/stage5_selection/20260629_stage5_mainline_full552_promotion_plan.sh
```

The full552 promotion shell is generated but not executed by the master screen
plan. Run it only after inspecting the limit64 selection JSON.

Counted hybrid condition-control continuation:

```text
docs/research/design_decisions/049_stage5_counted_hybrid_condition_control.md

scripts/eval_stage4_cod_lite_adapter.py:
  --counted-control-mode condition_residual_hybrid_affine_dct_grid
  --control-hybrid-selector-bytes
  --control-hybrid-rate-lambda

scripts/sweep_stage5_counted_control.py:
  mode=hybrid_affine_dct_grid
  selector_bytes=<int>
  rd_lambda=<float>
```

This is a per-image counted control selector over:

```text
none
affine
affine + DCT
affine + grid
```

The selector byte is counted in `actual_payload_bpp`. The generated screen is:

```text
results/stage5_counted_control/20260629_detailfilm_hybrid_limit64_sweep_plan.json
results/stage5_counted_control/20260629_detailfilm_hybrid_limit64_sweep_plan.sh
```

Planned conservative control bpp:

```text
0.000762939453125 to 0.002227783203125 at 512x512
```

It is now included in:

```text
results/stage5_counted_control/20260629_stage5_mainline_screen_select_plan.sh
```

CPU verification after hybrid extension:

```text
tests/test_eval_stage4_cod_lite_adapter_control.py
tests/test_sweep_stage5_counted_control.py

25 passed
```

Counted hybrid-basis condition-control continuation:

```text
docs/research/design_decisions/050_stage5_counted_hybrid_basis_condition_control.md

scripts/eval_stage4_cod_lite_adapter.py:
  --counted-control-mode condition_residual_hybrid_affine_dct_grid_basis

scripts/sweep_stage5_counted_control.py:
  mode=hybrid_affine_dct_grid_basis
  --override-mode hybrid_affine_dct_grid_basis
```

The post-affine basis prepare plan now also generates:

```text
results/stage5_counted_control/20260629_detailfilm_hybridbasis_limit64_sweep_plan.json
results/stage5_counted_control/20260629_detailfilm_hybridbasis_limit64_sweep_plan.sh
```

This lets the encoder choose per image among:

```text
none
affine
affine + DCT
affine + grid
affine + basis
```

The master screen now runs the generated hybrid-basis sweep and includes it in
cross-family anchor-aware selection.

CPU verification after hybrid-basis extension:

```text
tests/test_eval_stage4_cod_lite_adapter_control.py
tests/test_sweep_stage5_counted_control.py

28 passed
```

Hybrid image-RDO selector continuation:

```text
docs/research/design_decisions/051_stage5_hybrid_image_rdo_selector.md

scripts/eval_stage4_cod_lite_adapter.py:
  --control-hybrid-selection-objective condition_l1
  --control-hybrid-selection-objective image_l1
  --control-hybrid-selection-objective image_mse
  --control-hybrid-selection-objective lpips_alex
  --control-hybrid-selection-objective dists

scripts/sweep_stage5_counted_control.py:
  objective=<condition_l1|image_l1|image_mse|lpips_alex|dists>
```

Generated non-basis image-RDO screen:

```text
results/stage5_counted_control/20260629_detailfilm_hybrid_imagerdo_limit64_sweep_plan.json
results/stage5_counted_control/20260629_detailfilm_hybrid_imagerdo_limit64_sweep_plan.sh
```

The post-affine basis prepare plan now also generates:

```text
results/stage5_counted_control/20260629_detailfilm_hybridbasis_distsrdo_limit64_sweep_plan.json
results/stage5_counted_control/20260629_detailfilm_hybridbasis_distsrdo_limit64_sweep_plan.sh
```

CPU verification after image-RDO extension:

```text
tests/test_eval_stage4_cod_lite_adapter_control.py
tests/test_sweep_stage5_counted_control.py

29 passed
```

Candidate selection was added as the next step after a limit64 sweep:

```bash
.venv/bin/python scripts/select_stage5_control_candidates.py \
  --sweep-plan results/stage5_counted_control/20260629_bpp0312_limit64_sweep_plan.json \
  --reference-curve docs/research/baselines/cod_lite_kodak512_official_curve.json \
  --reference-metric lpips --reference-metric dists \
  --metric lpips --metric dists --metric psnr --metric ms_ssim \
  --min-basis-retained-energy 0.50 \
  --max-basis-residual-energy 0.50 \
  --top-k 8 \
  --output-json results/stage5_counted_control/20260629_bpp0312_limit64_candidate_selection.json
```

It ranks executed summaries by actual payload bpp, Pareto status, perceptual
gain, Stage3 fidelity guards, retained/residual basis energy guards when
available, and clearly labeled official CoD-Lite same-bpp gap diagnostics
before full552 promotion. Guard-failing rows remain in `candidates` for audit,
but are excluded from `recommended`.

Promotion command generation was added after candidate selection:

```bash
.venv/bin/python scripts/promote_stage5_control_candidates.py \
  --selection results/stage5_counted_control/20260629_bpp0312_basis_limit64_candidate_selection.json \
  --sweep-plan results/stage5_counted_control/20260629_bpp0312_basis_limit64_sweep_plan.json \
  --top-k 4 \
  --limit 0 \
  --batch-size 4 \
  --num-workers 4 \
  --run-suffix _full552 \
  --output-json results/stage5_counted_control/20260629_bpp0312_full552_promotion_commands.json \
  --output-sh results/stage5_counted_control/20260629_bpp0312_full552_promotion_commands.sh
```

This only rewrites selected limit64 eval commands into full552 commands. It does
not execute evaluation and does not change the selected control settings by
hand.

After running the generated full552 commands, collect curve points directly
from the promotion JSON:

```bash
.venv/bin/python scripts/collect_bd_curve_points.py \
  --promotion-json results/stage5_counted_control/20260629_bpp0312_full552_promotion_commands.json \
  --name coserdic_stage5_bpp0312_control_curve \
  --dataset gencodec512_full552 \
  --stage stage4 \
  --output-json results/bd_curves/coserdic_stage5_bpp0312_control_curve.json
```

If patch-FID/KID was computed as a separate JSON, add it with
`--extra-metric-json label=...` and request `--metric fid --metric kid`.

Then compute BD-rate only when the reference and candidate protocol match:

```bash
.venv/bin/python scripts/compute_bd_rate.py \
  --reference docs/research/baselines/cod_lite_kodak512_official_curve.json \
  --candidate results/bd_curves/coserdic_stage5_bpp0312_control_curve.json \
  --metric lpips --metric dists \
  --output-json results/bd_curves/coser_stage5_vs_cod_lite_bdrate.json
```

For a promotion gate on protocol-matched curves, add
`--fail-on-dataset-mismatch` so accidental Kodak/full552 mixing fails loudly.
The Kodak512 official curve is still useful as an engineering reference, but a
full552 candidate against Kodak512 reference is not a paper-valid BD-rate
claim.

Official CoD-Lite full552 baseline reproduction was planned but not executed
because CUDA/NVML is still unavailable:

```text
plan:
  results/baselines/cod_lite_official/20260629_cod512_full552_plan.json

shell:
  results/baselines/cod_lite_official/20260629_cod512_full552_plan.sh

doc:
  docs/research/baselines/cod_lite_full552_official_plan_20260629.md
```

The generated split manifests have the expected counts:

```text
full552: 552
kodak: 24
clic2020_test: 428
div2k_val: 100
```

After GPU restart, run:

```bash
bash results/baselines/cod_lite_official/20260629_cod512_full552_plan.sh
```

This evaluates official CoD-Lite at 0.0039/0.0078/0.0156/0.0312 bpp. The
full552 aggregate run uses `fid_patch_size=0` for PSNR/MS-SSIM/LPIPS/DISTS,
while split runs compute protocol-specific FID: Kodak patch64 and
CLIC2020/DIV2K patch256.

Official CoD one-step full552 baseline reproduction was also planned:

```text
plan:
  results/baselines/cod_one_step_official/20260629_cod512_full552_plan.json

shell:
  results/baselines/cod_one_step_official/20260629_cod512_full552_plan.sh

doc:
  docs/research/baselines/cod_one_step_full552_official_plan_20260629.md
```

After GPU restart, run:

```bash
bash results/baselines/cod_one_step_official/20260629_cod512_full552_plan.sh
```

This evaluates official one-step CoD at 0.0039/0.0312/0.1250 bpp with the same
full552/split FID protocol split.

## Existing Stage 4 Full552 CPU Audit

A CPU-only scan of existing `results/stage4_cod_lite_adapter_eval/*/summary.json`
found 37 full552 summaries. Current best internal Stage 4 anchors at unchanged
CoSER actual payload bpp are:

```text
best LPIPS:
  run: 20260629_stage4_fast8192_detailfilm_ft800_b8_full552_eval
  bpp: 0.013999109682829483
  PSNR / MS-SSIM: 21.2427 / 0.7150
  LPIPS / DISTS: 0.4304 / 0.2982

best DISTS among the inspected top rows:
  run: 20260628_stage4_detailaware_adapter_bpp0312_transfer_ft600_b2_full552_eval
  bpp: 0.013999109682829483
  PSNR / MS-SSIM: 21.4438 / 0.7216
  LPIPS / DISTS: 0.4402 / 0.2917

Stage 3 anchor in these summaries:
  LPIPS / DISTS: 0.5758 / 0.3536
```

Interpretation:

```text
Stage 4 has real signal over Stage 3 at unchanged actual_payload_bpp.
It is still not an external-baseline win.
The next Stage 5 probes should beat the best full552 Stage 4 anchors first,
then be compared against the official CoD-Lite full552 curves once the GPU
baseline plan has run.
```

## GPU-Restart Next Commands

Inspect CoD-Lite parameter names before choosing LoRA regexes. This was run
successfully on CPU because CUDA was unavailable:

```bash
.venv/bin/python scripts/inspect_cod_lite_backbone_params.py \
  --device cpu \
  --cod-lite-checkpoint external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0312.pt \
  --cod-lite-config external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0312.yaml \
  --pattern 'final|dec|decoder|output|attn|mlp|proj|ffn|block' \
  --lora-rank 4 \
  --output-json results/stage5_inspect/20260629_cod_lite_bpp0312_lora_candidate_params_cpu.json
```

Generate LoRA target presets:

```bash
.venv/bin/python scripts/plan_stage5_lora_targets.py \
  --inspect-json results/stage5_inspect/20260629_cod_lite_bpp0312_lora_candidate_params_cpu.json \
  --preset denoiser_tail \
  --preset dec_net \
  --preset denoiser_mlp \
  --preset denoiser_all \
  --preset y_decoder \
  --last-blocks 6 \
  --output-json results/stage5_inspect/20260629_cod_lite_bpp0312_lora_target_plan_rank4.json
```

Generated rank-4 candidate sizes:

```text
denoiser_tail:
  30 modules / 147,456 LoRA params

dec_net:
  11 modules / 37,632 LoRA params

denoiser_mlp:
  42 modules / 322,560 LoRA params

denoiser_all:
  105 modules / 516,096 LoRA params

y_decoder:
  16 modules / 138,240 LoRA params
  riskier because it touches the condition-codec decoder
```

Generate train + limit64 eval commands for the first conservative LoRA probes:

```bash
.venv/bin/python scripts/sweep_stage5_lora_backbone.py \
  --lora-plan results/stage5_inspect/20260629_cod_lite_bpp0312_lora_target_plan_rank4.json \
  --preset denoiser_tail \
  --preset dec_net \
  --run-prefix 20260629_stage5_bpp0312_lora \
  --train-manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl \
  --eval-manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl \
  --eval-per-image-metrics results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/per_image_metrics.jsonl \
  --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_detailaware_adapter_bpp0312_transfer_ft600_b2.pt \
  --max-steps 1000 \
  --output-json results/stage5_lora/20260629_bpp0312_lora_denoiser_tail_decnet_limit64_plan.json \
  --output-sh results/stage5_lora/20260629_bpp0312_lora_denoiser_tail_decnet_limit64_plan.sh
```

After GPU restart, run:

```bash
bash results/stage5_lora/20260629_bpp0312_lora_denoiser_tail_decnet_limit64_plan.sh
```

Fit basis on non-eval train cache:

```bash
.venv/bin/python scripts/fit_stage5_condition_control_basis.py \
  --checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_detailaware_adapter_bpp0312_transfer_ft600_b2.pt \
  --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl \
  --run-name 20260629_stage5_bpp0312_condition_basis_g16_s8_k64_train8192 \
  --crop-size 512 --batch-size 4 --num-workers 4 \
  --groups 16 --grid-size 8 --components 64 \
  --huffman-bits 4 --huffman-quantile p95 --huffman-quantile p99 \
  --huffman-quantizer uniform --huffman-quantizer mu_law --huffman-mu 16 \
  --sparse-topk-candidate-components 64 \
  --sparse-topk-components 8 16 32
```

Plan multi-rate control settings for BD-rate preparation:

```bash
.venv/bin/python scripts/plan_stage5_control_curve.py \
  --settings-json outputs/stage5_control_basis/20260629_stage5_bpp0312_condition_basis_g16_s8_k64_train8192/recommended_basis_settings.json \
  --band 0:0.00025:2 \
  --band 0.00025:0.001:2 \
  --band 0.001:0.002:2 \
  --rank-by retained_per_bpp \
  --include-selection prefix --include-selection topk \
  --include-codec fixed_bits --include-codec huffman \
  --include-quantizer uniform --include-quantizer mu_law \
  --min-basis-retained-energy 0.50 \
  --max-basis-residual-energy 0.50 \
  --max-quantization-rmse 0.20 \
  --max-clipped-fraction 0.10 \
  --output-json results/stage5_counted_control/20260629_bpp0312_curve_settings.json
```

Then run the limit64 counted-control sweep:

```bash
.venv/bin/python scripts/sweep_stage5_counted_control.py \
  --checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_detailaware_adapter_bpp0312_transfer_ft600_b2.pt \
  --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl \
  --per-image-metrics results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/per_image_metrics.jsonl \
  --control-basis outputs/stage5_control_basis/20260629_stage5_bpp0312_condition_basis_g16_s8_k64_train8192/control_basis.pt \
  --run-prefix 20260629_stage5_bpp0312_counted_control_limit64 \
  --settings-json results/stage5_counted_control/20260629_bpp0312_curve_settings.json \
  --setting mode=dct,groups=16,grid=8,coeffs=4,bits=4,range=0.25,scale=1.0 \
  --setting mode=dct,groups=16,grid=8,coeffs=8,bits=4,range=0.25,scale=1.0 \
  --setting mode=grid,groups=8,grid=4,bits=4,range=0.25,scale=1.0 \
  --output-json results/stage5_counted_control/20260629_bpp0312_limit64_sweep_plan.json
```

After basis fitting, adjust basis `range` from the fit summary p95/p99 before
using the sweep for promotion decisions.

Inspect the fitted basis before choosing ranges:

```bash
.venv/bin/python scripts/inspect_stage5_control_basis.py \
  --basis outputs/stage5_control_basis/20260629_stage5_bpp0312_condition_basis_g16_s8_k64_train8192/control_basis.pt \
  --components 8 16 32 64 \
  --candidate-components 64 \
  --selection prefix --selection topk \
  --quantile p95 --quantile p99 \
  --bits 4 \
  --codec both \
  --quantizer uniform --quantizer mu_law --mu 16 \
  --output-json outputs/stage5_control_basis/20260629_stage5_bpp0312_condition_basis_g16_s8_k64_train8192/recommended_basis_settings.json
```

## Condition Residual Guard Continuation

This CPU continuation added a deterministic decoder-side condition residual
RMS guard:

```text
base_condition = CoD-Lite native condition from decoded Stage 3 x_aux
condition_residual = adapter_condition - base_condition
guarded_residual = rms_clipped(condition_residual, base_condition)
```

This is not RGB postprocessing and does not add transmitted bits. It is a
Stage 5 stability probe for condition recovery.

Added / updated:

```text
src/coserdic/models/gencodec_backbone.py
  condition_residual_rms_guard

scripts/eval_stage4_cod_lite_adapter.py
  --condition-residual-guard none|rms_clip
  --condition-residual-guard-granularity global|spatial|channel
  --condition-residual-max-rms-ratio
  --condition-residual-min-gate

scripts/sweep_stage5_condition_residual_guard.py
scripts/select_stage5_control_candidates.py
```

Generated:

```text
results/stage5_condition_guard/20260629_detailfilm_guard_screen64_plan.json
results/stage5_condition_guard/20260629_detailfilm_guard_screen64_plan.sh
results/stage5_condition_guard/20260629_detailfilm_guard_full552_plan.json
results/stage5_condition_guard/20260629_detailfilm_guard_full552_plan.sh
```

The sweep covers:

```text
guard_none
global/channel/spatial RMS ratios:
  0.15, 0.25, 0.35, 0.50, 0.75, 1.00
```

Recommended GPU-restart command:

```bash
bash results/stage5_condition_guard/20260629_detailfilm_guard_screen64_plan.sh
```

Run full552 only for screen64 candidates that beat `guard_none` without
fidelity collapse.

Additional verification:

```text
py_compile:
  scripts/eval_stage4_cod_lite_adapter.py
  scripts/sweep_stage5_condition_residual_guard.py
  scripts/select_stage5_control_candidates.py
  src/coserdic/models/gencodec_backbone.py

related tests:
  tests/test_gencodec_backbone.py
  tests/test_sweep_stage5_condition_residual_guard.py
  tests/test_select_stage5_control_candidates.py

result:
  32 passed
```

## Guarded LoRA Training Plan

The training script now supports optional condition residual RMS-excess
regularization:

```text
condition_residual_rms_guard_loss =
  mean(relu(residual_rms / base_rms - max_ratio)^2)
```

This is a training-time condition-space stabilizer, not an evaluation-time RGB
blend. Default weight is `0.0`.

Generated a new current-anchor Stage 5 LoRA plan from:

```text
checkpoints/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailfilm_ft800_b8.pt
```

Plan:

```text
results/stage5_lora/20260629_detailfilm_lora_guarded_denoiser_tail_decnet_limit64_plan.json
results/stage5_lora/20260629_detailfilm_lora_guarded_denoiser_tail_decnet_limit64_plan.sh
```

Settings:

```text
denoiser_tail:
  30 modules
  147,456 rank-4 LoRA params

dec_net:
  11 modules
  37,632 rank-4 LoRA params

condition residual RMS regularization:
  weight: 0.05
  ratio: 0.35
  granularity: channel
```

Recommended GPU-restart command:

```bash
bash results/stage5_lora/20260629_detailfilm_lora_guarded_denoiser_tail_decnet_limit64_plan.sh
```

Verification:

```text
py_compile:
  scripts/train_stage4_cod_lite_adapter.py
  scripts/sweep_stage5_lora_backbone.py
  tests/test_train_stage4_cod_lite_adapter_regularization.py
  tests/test_sweep_stage5_lora_backbone.py

related tests:
  8 passed
```

## Anchor-Aware Selection Continuation

The candidate selector now supports comparing Stage 5 candidates against a
current internal anchor, not only against Stage 3:

```text
scripts/select_stage5_control_candidates.py
  --anchor-summary
  --anchor-label
  --max-anchor-psnr-drop
  --max-anchor-ms-ssim-drop
  --max-anchor-lpips-increase
  --max-anchor-dists-increase
```

Rows now include:

```text
*_delta_vs_anchor
*_improves_vs_anchor
anchor_guard_pass
```

CPU analysis output:

```text
results/stage5_selection/20260629_current_full552_vs_detailfilm_anchor_psnr08.json
```

Anchor:

```text
results/stage4_cod_lite_adapter_eval/20260629_stage4_fast8192_detailfilm_ft800_b8_full552_eval/summary.json
```

Result:

```text
loaded existing full552 Stage 4 candidates: 6
recommended under strict LPIPS/DISTS anchor guard: only the detail-FiLM anchor
```

This confirms that existing Stage 4 variants do not beat the current anchor
and that Stage 5 LoRA / guard / counted-control candidates should be selected
with anchor-aware promotion before full552 runs.

Verification:

```text
py_compile:
  scripts/select_stage5_control_candidates.py
  tests/test_select_stage5_control_candidates.py

related tests:
  tests/test_select_stage5_control_candidates.py

result:
  9 passed
```

## Component Codebook Basis Quantization Continuation

Added fixed decoder-side component codebooks for Stage 5 basis coefficients.

This extends the component-wise p95/p99 range path:

```text
component_p95 / component_p99:
  per-component fixed ranges
  uniform or mu-law levels inside each range

component_codebook:
  per-component fixed scalar Lloyd codebook
  fixed-bit level indices only are transmitted per image
```

The codebooks are fitted from non-eval train-cache coefficient distributions
and stored in the basis checkpoint:

```text
coefficient_component_codebooks:
  lloyd_b4:
    codec: component_codebook_control_fixed_bits
    method: lloyd_1d_per_component
    bits / levels / component_count
    fixed_bytes_per_image
    quantization_mae / quantization_rmse / max_abs_error
    codebooks
```

New CLI hooks:

```text
scripts/fit_stage5_condition_control_basis.py
  --component-codebook-bits 4
  --component-codebook-max-iter 50

scripts/inspect_stage5_control_basis.py
  --range-mode component_codebook

scripts/eval_stage4_cod_lite_adapter.py
  --control-basis-range-mode component_codebook
```

The GPU-restart post-affine basis prepare plan now includes:

```text
--component-codebook-bits 4
--component-codebook-max-iter 50
--range-mode component_codebook
```

This is still counted-control mainline infrastructure. It does not transmit
per-image codebooks and does not modify RGB output after decoding.

Verification:

```bash
PYTHONPATH=/workspace/CoSER-DiC .venv/bin/pytest \
  tests/test_control_grid.py \
  tests/test_fit_stage5_condition_control_basis.py \
  tests/test_inspect_stage5_control_basis.py \
  tests/test_sweep_stage5_counted_control.py \
  tests/test_eval_stage4_cod_lite_adapter_control.py -q
```

Result:

```text
64 passed
```

## Component Codebook Huffman Continuation

Added static Huffman coding for component-codebook basis-control symbols.

New fit-time priors:

```text
control_huffman_priors:
  p95_b4_codebook
  p99_b4_codebook

sparse_topk_control_priors:
  topk_c{C}_k{K}_p95_b4_codebook
  topk_c{C}_k{K}_p99_b4_codebook
```

These priors store:

```text
basis_range_mode: component_codebook
codebook_key: lloyd_b4
mean/min/max payload bytes
prefix_mean_payload_bytes
static Huffman tables
```

`inspect_stage5_control_basis.py` now emits codebook Huffman settings when the
matching priors exist:

```text
codec=huffman
basis_range_mode=component_codebook
huffman_key=p95_b4_codebook
huffman_key=topk_c64_k8_p95_b4_codebook
```

At evaluation, the Huffman payload is decoded as codebook level indices and
then dequantized with `ComponentCodebookControlCode`. The exact Huffman bytes
are counted per image in `actual_payload_bpp`.

Verification:

```bash
PYTHONPATH=/workspace/CoSER-DiC .venv/bin/pytest \
  tests/test_fit_stage5_condition_control_basis.py \
  tests/test_inspect_stage5_control_basis.py \
  tests/test_sweep_stage5_counted_control.py \
  tests/test_eval_stage4_cod_lite_adapter_control.py \
  tests/test_control_grid.py -q
```

Result:

```text
69 passed
```

## Vector Codebook Basis Control Continuation

Added a low-bpp vector-codebook basis-control family.

Instead of sending K scalar basis coefficients, this path sends one codebook
index for the whole K-dimensional coefficient prefix:

```text
selection=vector
fixed decoder-side vector codebook
per-image payload:
  one fixed-bit vector index
  or one static-Huffman-coded vector index
```

New fit hooks:

```text
--vector-codebook-components 4 8 16
--vector-codebook-bits 4 6 8
--vector-codebook-max-iter 50
```

New eval/sweep hook:

```text
--control-basis-selection vector
```

Fit output:

```text
coefficient_vector_codebooks:
  vq_k8_b8:
    codec: vector_codebook_control_fixed_bits
    selection: vector
    fixed_bytes_per_image
    huffman_mean_payload_bytes
    quantization_mae / quantization_rmse
    vectors
    huffman
```

The post-affine basis prepare plan now includes vector fitting and inspect emits
`selection=vector` settings. These are intended as very-low-control-bpp curve
points and hybrid candidates, not as a replacement for scalar/top-k basis
controls.

Also fixed the direct `condition_residual_basis` eval path to pass
`basis_control_codec` instead of the scalar `control_codec`, so component
ranges/codebooks and vector codebooks are actually used in direct basis mode.

Verification:

```bash
PYTHONPATH=/workspace/CoSER-DiC .venv/bin/pytest \
  tests/test_control_grid.py \
  tests/test_fit_stage5_condition_control_basis.py \
  tests/test_inspect_stage5_control_basis.py \
  tests/test_sweep_stage5_counted_control.py \
  tests/test_eval_stage4_cod_lite_adapter_control.py -q
```

Result:

```text
75 passed
```

## Compact Sparse Top-k Payload Continuation

Fixed-bit sparse top-k basis controls now pack the selected component indices
and their quantized values into one compact payload stream.

Previous conservative simulation:

```text
index fixed-bit stream rounded to bytes
+ value fixed-bit stream rounded to bytes
```

Current fixed-bit payload:

```text
ceil((topk * index_bits + topk * value_bits) / 8)
```

This does not change the transmitted information. It only removes unnecessary
extra byte rounding for a deterministic fixed-bit stream.

Example covered by tests:

```text
candidate_components=3
topk=1
index_bits=2
value_bits=2

compact payload:
  4 bits -> 1 byte
```

Updated paths:

```text
src/coserdic/entropy/control_grid.py
  SparseControlBasisCode.encode_compact
  SparseControlBasisCode.decode_compact
  encoded_compact_num_bits / encoded_compact_num_bytes

scripts/eval_stage4_cod_lite_adapter.py
  fixed_bits top-k basis eval uses compact payload
  hybrid-basis exact selector bits use compact top-k bit counts

scripts/sweep_stage5_counted_control.py
  dry-run control_bytes uses compact top-k fixed-bit payloads
```

Huffman sparse top-k stays as exact index-Huffman + value-Huffman bytes because
those are real variable-length entropy-coded streams.

Verification:

```bash
PYTHONPATH=/workspace/CoSER-DiC .venv/bin/pytest \
  tests/test_control_grid.py \
  tests/test_eval_stage4_cod_lite_adapter_control.py \
  tests/test_sweep_stage5_counted_control.py -q
```

Result:

```text
67 passed
```

Full suite:

```bash
PYTHONPATH=/workspace/CoSER-DiC .venv/bin/pytest tests -q
```

Result:

```text
263 passed, 4 warnings
```

GPU status remains blocked:

```text
nvidia-smi: Failed to initialize NVML: Unknown Error
```
