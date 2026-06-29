# BD-Rate Tooling and Current External Gap

Date: 2026-06-29

## Decision

Add explicit BD-rate tooling now, before claiming any Stage 5 improvement.

The current CoSER-DiC Stage 4 results are not BD-rate competitive against
official CoD-Lite. On Kodak512, current CoSER perceptual metrics do not even
overlap the official CoD-Lite LPIPS/DISTS curve range.

This is a hard negative result, but it is useful:

```text
Stage 5 cannot be claimed from one operating point
Stage 5 cannot be claimed while perceptual metric ranges do not overlap
the next model work must make a large perceptual jump, not only small adapter
cleanups
```

## Implementation

Added:

```text
scripts/compute_bd_rate.py
scripts/collect_bd_curve_points.py
tests/test_compute_bd_rate.py
tests/test_collect_bd_curve_points.py
docs/research/baselines/cod_lite_kodak512_official_curve.json
```

`compute_bd_rate.py` computes Bjontegaard delta-rate by integrating log(rate)
over the overlapping metric range with piecewise-linear interpolation. It
refuses:

```text
one-point candidate curves
one-point reference curves
non-overlapping metric ranges
```

This is intentional. It prevents accidental BD-rate claims from single-point
or off-range comparisons.

`collect_bd_curve_points.py` converts CoSER summary files into curve JSON:

```text
summary.json
split_summary_cod512.json
```

It supports:

```text
stage3 summary fields
stage4 summary fields
direct baseline fields
dataset split extraction from summarize_per_image_metrics output
```

## Official CoD-Lite Kodak512 Curve

Stored as:

```text
docs/research/baselines/cod_lite_kodak512_official_curve.json
```

Current points:

| bpp | LPIPS | DISTS | FID | PSNR | MS-SSIM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00390625 | 0.3390 | 0.2089 | 51.2190 | n/a | n/a |
| 0.00781250 | 0.2807 | 0.1733 | 44.6564 | n/a | n/a |
| 0.01562500 | 0.2259 | 0.1402 | 38.4785 | 20.7667 | 0.7090 |
| 0.03125000 | 0.1614 | 0.1120 | 31.8072 | n/a | n/a |

Important:

```text
these are official nominal bpp values from the CoD-Lite rate checkpoints
CoSER paper tables should use actual_payload_bpp
do not mix datasets or bpp definitions without labeling
```

## Current CoSER Kodak Probe

Generated:

```text
results/bd_curves/coser_stage4_current_two_point_mixed_prior_kodak.json
results/bd_curves/coser_vs_cod_lite_kodak_bd_rate_probe_20260629.json
```

Current Kodak points:

| point | actual_payload_bpp | PSNR | MS-SSIM | LPIPS | DISTS |
| --- | ---: | ---: | ---: | ---: | ---: |
| detail-FiLM bpp0.0156 prior | 0.014123 | 20.9936 | 0.6997 | 0.4678 | 0.3169 |
| bpp0.0312 transfer prior | 0.014123 | 21.1635 | 0.7050 | 0.4827 | 0.3126 |

BD-rate attempt:

| metric | status | reason |
| --- | --- | --- |
| LPIPS | no_metric_overlap | current best CoSER Kodak LPIPS is 0.4678; official CoD-Lite worst point is 0.3390 |
| DISTS | no_metric_overlap | current best CoSER Kodak DISTS is 0.3126; official CoD-Lite worst point is 0.2089 |
| PSNR | insufficient_points | official CoD-Lite Kodak PSNR is currently stored for only one rate point |

Interpretation:

```text
CoSER is not merely losing BD-rate;
for LPIPS/DISTS it is outside the official CoD-Lite curve range
```

## Research Consequence

Small condition-adapter improvements are not enough.

The next model moves must target a much larger perceptual gain:

```text
stronger CoD-Lite / CoD condition recovery
rate-specific bpp0.0312 Detail-FiLM transfer after GPU restart
more aggressive perceptual condition adapter objective, with fidelity guard
partial unfreeze / LoRA after adapter saturation
diffusion-control detail head with channel/group targets
eventual tiny counted control stream if deterministic decoder-side control saturates
```

The first concrete partial-unfreeze implementation and restart command are in:

```text
docs/research/design_decisions/038_stage5_partial_backbone_finetune_path.md
```

The first concrete tiny counted control-stream implementation and restart
sweep are in:

```text
docs/research/design_decisions/039_stage5_tiny_counted_condition_control_stream.md
```

The concrete short-term target is:

```text
Kodak512 LPIPS below 0.339
Kodak512 DISTS below 0.2089
```

This is the minimum needed before LPIPS/DISTS BD-rate against the stored
official CoD-Lite Kodak curve can even be computed.

## Usage

Collect CoSER curve points from full552 overall summaries:

```bash
.venv/bin/python scripts/collect_bd_curve_points.py \
  --input label_a=path/to/summary_a.json \
  --input label_b=path/to/summary_b.json \
  --name coser_curve_name \
  --dataset strict512_full552 \
  --stage stage4 \
  --output-json results/bd_curves/coser_curve_name.json
```

Collect Kodak split points:

```bash
.venv/bin/python scripts/collect_bd_curve_points.py \
  --input label_a=path/to/split_summary_cod512.json \
  --input label_b=path/to/split_summary_cod512.json \
  --name coser_kodak_curve_name \
  --dataset Kodak512 \
  --stage stage4 \
  --dataset-key kodak \
  --output-json results/bd_curves/coser_kodak_curve_name.json
```

Compute BD-rate:

```bash
.venv/bin/python scripts/compute_bd_rate.py \
  --reference docs/research/baselines/cod_lite_kodak512_official_curve.json \
  --reference-name official_cod_lite_kodak512 \
  --candidate results/bd_curves/coser_kodak_curve_name.json \
  --candidate-name coser_candidate \
  --metric lpips --metric dists \
  --output-json results/bd_curves/coser_vs_cod_lite_bd_rate.json
```

Use `--fail-on-unavailable` in automated promotion gates so that
insufficient-point or non-overlap cases cannot pass silently.

Use `--max-bd-rate-percent` for Stage 5 promotion thresholds:

```bash
.venv/bin/python scripts/compute_bd_rate.py \
  --reference docs/research/baselines/cod_lite_kodak512_official_curve.json \
  --candidate results/bd_curves/coser_kodak_curve_name.json \
  --metric lpips --metric dists \
  --max-bd-rate-percent -10 \
  --output-json results/bd_curves/stage5_promotion_gate.json
```

This exits nonzero if any requested metric is unavailable or worse than the
threshold. For Stage 5, this prevents promoting a run that merely improves
Stage 3 internally but does not beat the external curve.

## GPU Status

The bpp0.0312 Detail-FiLM continuation remains paused because GPU access is
still unavailable:

```text
nvidia-smi: Failed to initialize NVML: Unknown Error
torch.cuda.is_available: False
```

Per project policy, no GPU experiment should be launched until the container is
restarted.
