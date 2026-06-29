# Decision 048: Stage 5 Counted-Control Master Screen and Selection

Date: 2026-06-29 JST

## Status

Accepted as the current GPU-restart path for Stage 5 counted-control probes.

GPU execution is still blocked on the current container because CUDA/NVML is
not visible. The plan below is prepared, syntax-checked, and documented, but
not executed.

## Problem

Stage 5 counted-control work now has several legitimate same-family control
routes:

```text
condition_residual_affine
condition_residual_affine_dct
condition_residual_affine_grid
condition_residual_affine_basis
condition_residual_hybrid_affine_dct_grid
condition_residual_hybrid_affine_dct_grid_basis
```

All of them modify the CoD-Lite condition space using explicitly counted
image-specific payload bytes. This is mainline CoSER-DiC information flow, not
RGB postprocessing.

The risk is that the research loop becomes hand-picked:

```text
run one family
look at a few rows
promote the most convenient setting
```

That would make Stage 5 promotion fragile and difficult to audit.

## Decision

Use one master screen/select plan as the preferred entrypoint:

```bash
bash results/stage5_counted_control/20260629_stage5_mainline_screen_select_plan.sh
```

The shell plan runs:

```text
1. affine curve-oriented limit64 screen
2. affine+DCT limit64 screen
3. affine+grid limit64 screen
4. hybrid none/affine/affine+DCT/affine+grid limit64 screen
5. image-RDO hybrid none/affine/affine+DCT/affine+grid limit64 screen
6. post-affine basis fit on non-eval train cache
7. generated affine+basis limit64 screen
8. generated hybrid-basis limit64 screen
9. generated DISTS-RDO hybrid-basis limit64 screen
10. cross-family anchor-aware candidate selection
11. full552 promotion-plan generation for selected candidates
```

The plan starts with:

```bash
.venv/bin/python scripts/check_gpu_ready.py --min-devices 1
```

so it fails before launching long GPU work when CUDA/NVML is unavailable.

## Anchor and Promotion Policy

Selection must compare candidates against the current detail-FiLM limit64
anchor:

```text
results/stage4_cod_lite_adapter_eval/20260629_stage4_fast8192_detailfilm_ft800_b8_limit64_eval/summary.json
```

The master selection output is:

```text
results/stage5_selection/20260629_stage5_mainline_limit64_all_control_selection.json
```

The generated full552 promotion artifacts are:

```text
results/stage5_selection/20260629_stage5_mainline_full552_promotion_plan.json
results/stage5_selection/20260629_stage5_mainline_full552_promotion_plan.sh
```

The master plan intentionally does not run the full552 promotion shell. The
selection JSON must be inspected first. Full552 is expensive and should only be
spent on candidates that pass the limit64 anchor guards.

## Bpp Policy

For every counted-control mode:

```text
paper_bpp = actual_payload_bpp
actual_payload_bpp = Stage 3 semantic/detail payload bpp + control_payload_bpp
```

All per-image gain, bias, DCT, grid, basis coefficient, sparse index, and
Huffman-coded value bytes are counted. Fixed decoder-side state such as the
trained basis, static Huffman priors, mu-law parameter, model weights, and
checkpoint config is not counted as image payload.

## Non-Goals

This decision does not claim a Stage 5 win.

It does not:

```text
use final RGB alpha blending as the method
use uncounted image-specific side information
replace CoSER semantic/detail streams with CoD-Lite native streams
claim BD-rate from a limit64 screen
compare protocol-mismatched curves as paper evidence
```

## Next Actions After GPU Restart

Run:

```bash
bash results/stage5_counted_control/20260629_stage5_mainline_screen_select_plan.sh
```

Then inspect:

```text
results/stage5_selection/20260629_stage5_mainline_limit64_all_control_selection.json
```

If guarded candidates are recommended, run:

```bash
bash results/stage5_selection/20260629_stage5_mainline_full552_promotion_plan.sh
```

Only after full552 candidates are evaluated should CoSER curve points be
collected and compared against protocol-matched official baselines.
