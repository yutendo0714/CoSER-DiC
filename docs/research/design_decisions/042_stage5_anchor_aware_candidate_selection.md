# 042 Stage 5 Anchor-Aware Candidate Selection

Date: 2026-06-29

## Decision

Stage 5 candidate selection must compare each new candidate not only against
Stage 3, but also against the current strongest Stage 4 / Stage 5 anchor.

The active anchor for no-extra-payload CoD-Lite adapter work is:

```text
limit64 screening anchor:
  20260629_stage4_fast8192_detailfilm_ft800_b8_limit64_eval

  actual_payload_bpp: 0.013654232025146484
  PSNR: 21.6393
  MS-SSIM: 0.7389
  LPIPS: 0.4091
  DISTS: 0.2960

full552 reporting anchor:
20260629_stage4_fast8192_detailfilm_ft800_b8_full552_eval

  actual_payload_bpp: 0.013999109682829483
  PSNR: 21.2427
  MS-SSIM: 0.7150
  LPIPS: 0.4304
  DISTS: 0.2982
```

This anchor is still far behind official CoD-Lite, but it is the strongest
current internal branch for LPIPS/DISTS at unchanged CoSER semantic/detail
payload. Stage 5 promotion should therefore require a candidate to beat or
match the matching-split anchor under explicitly recorded guards. Do not compare
limit64 candidates directly against the full552 anchor.

## Implementation

Updated:

```text
scripts/select_stage5_control_candidates.py
```

Added:

```text
--anchor-summary <summary.json>
--anchor-label <loaded-candidate-label>
--fail-on-anchor-count-mismatch
--max-anchor-psnr-drop
--max-anchor-ms-ssim-drop
--max-anchor-lpips-increase
--max-anchor-dists-increase
```

Each candidate now records:

```text
actual_payload_bpp_delta_vs_anchor
psnr_delta_vs_anchor
ms_ssim_delta_vs_anchor
lpips_delta_vs_anchor
dists_delta_vs_anchor
condition_l1_delta_vs_anchor
*_improves_vs_anchor
anchor_count_match
count_delta_vs_anchor
anchor_guard_pass
```

`promotion_guard_pass` now requires:

```text
stage3_guard_pass
and basis_guard_pass, when applicable
and anchor_guard_pass, when an anchor is provided
```

This avoids promoting candidates that improve Stage 3 but merely regress from
the current best CoSER diffusion branch.

If both candidate and anchor summaries expose `count`, the selector records
`anchor_count_warnings` when counts differ. Use
`--fail-on-anchor-count-mismatch` for promotion runs.

## CPU Analysis

Ran an anchor-aware internal full552 selection:

```text
output:
  results/stage5_selection/20260629_current_full552_vs_detailfilm_anchor_psnr08.json

anchor:
  results/stage4_cod_lite_adapter_eval/20260629_stage4_fast8192_detailfilm_ft800_b8_full552_eval/summary.json

stage3 guard:
  max_psnr_drop: 0.8 dB

anchor guard:
  max_psnr_drop: 0.25 dB
  max_ms_ssim_drop: 0.01
  max_lpips_increase: 0.0
  max_dists_increase: 0.0
```

Result:

```text
loaded candidates: 6
recommended candidates: 1
recommended: current detail-FiLM anchor itself
```

Also ran a limit64 anchor sanity selection:

```text
output:
  results/stage5_selection/20260629_detailfilm_limit64_anchor_sanity.json

anchor:
  results/stage4_cod_lite_adapter_eval/20260629_stage4_fast8192_detailfilm_ft800_b8_limit64_eval/summary.json

strict count check:
  --fail-on-anchor-count-mismatch

loaded candidates:
  detail-FiLM normal / detail-zero / detail-shuffle limit64

recommended:
  current detail-FiLM limit64 anchor itself
```

Interpretation:

```text
Existing full552 Stage 4 branches do not beat the detail-FiLM anchor on the
strict LPIPS/DISTS anchor guard. Some branches improve DISTS or fidelity, but
not LPIPS at the same time. This reinforces that Stage 5 must produce a larger
condition/control/backbone improvement instead of sweeping shallow adapter
variants.
```

## Next Use

After GPU restart and limit64 execution, run selector with the matching
limit64 anchor:

```bash
.venv/bin/python scripts/select_stage5_control_candidates.py \
  --sweep-plan results/stage5_condition_guard/20260629_detailfilm_guard_screen64_plan.json \
  --anchor-summary results/stage4_cod_lite_adapter_eval/20260629_stage4_fast8192_detailfilm_ft800_b8_limit64_eval/summary.json \
  --fail-on-anchor-count-mismatch \
  --max-psnr-drop 0.8 \
  --max-anchor-psnr-drop 0.25 \
  --max-anchor-ms-ssim-drop 0.01 \
  --max-anchor-lpips-increase 0.0 \
  --max-anchor-dists-increase 0.0 \
  --output-json results/stage5_selection/20260629_guard_screen64_anchor_selection.json
```

For guarded LoRA:

```bash
.venv/bin/python scripts/select_stage5_control_candidates.py \
  --sweep-plan results/stage5_lora/20260629_detailfilm_lora_guarded_denoiser_tail_decnet_limit64_plan.json \
  --anchor-summary results/stage4_cod_lite_adapter_eval/20260629_stage4_fast8192_detailfilm_ft800_b8_limit64_eval/summary.json \
  --fail-on-anchor-count-mismatch \
  --max-psnr-drop 0.8 \
  --max-anchor-psnr-drop 0.25 \
  --max-anchor-ms-ssim-drop 0.01 \
  --max-anchor-lpips-increase 0.0 \
  --max-anchor-dists-increase 0.0 \
  --output-json results/stage5_selection/20260629_detailfilm_lora_guarded_anchor_selection.json
```

Then promote only recommended Pareto candidates to full552:

```bash
.venv/bin/python scripts/promote_stage5_control_candidates.py \
  --selection results/stage5_selection/20260629_detailfilm_lora_guarded_anchor_selection.json \
  --sweep-plan results/stage5_lora/20260629_detailfilm_lora_guarded_denoiser_tail_decnet_limit64_plan.json \
  --top-k 2 \
  --limit 0 \
  --batch-size 4 \
  --num-workers 4 \
  --output-json results/stage5_selection/20260629_detailfilm_lora_guarded_full552_promotion.json \
  --output-sh results/stage5_selection/20260629_detailfilm_lora_guarded_full552_promotion.sh
```

After promoted full552 runs finish, run selector again with the full552 anchor:

```bash
.venv/bin/python scripts/select_stage5_control_candidates.py \
  --sweep-plan results/stage5_selection/20260629_detailfilm_lora_guarded_full552_promotion.json \
  --anchor-summary results/stage4_cod_lite_adapter_eval/20260629_stage4_fast8192_detailfilm_ft800_b8_full552_eval/summary.json \
  --fail-on-anchor-count-mismatch \
  --max-psnr-drop 0.8 \
  --max-anchor-psnr-drop 0.25 \
  --max-anchor-ms-ssim-drop 0.01 \
  --max-anchor-lpips-increase 0.0 \
  --max-anchor-dists-increase 0.0 \
  --output-json results/stage5_selection/20260629_detailfilm_lora_guarded_full552_anchor_selection.json
```

Do not report BD-rate from these single candidates. Use them only to decide
which operating points deserve full552 evaluation and later multi-rate curve
construction.
