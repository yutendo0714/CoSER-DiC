# Stage 5 Tiny Counted Condition-Control Stream

Date: 2026-06-29

## Decision

Add a Stage 5 candidate path for a tiny entropy-coded condition-control stream.

This is a mainline information-flow change, not post-hoc output editing:

```text
decoded semantic/detail CoSER payload
  -> CoSER condition adapter
  -> CoD-Lite condition estimate
  + tiny counted condition-control payload
  -> CoD-Lite / CoD decoder backbone
```

The control stream is image-specific side information. Therefore it must be
included in:

```text
actual_payload_bpp / paper_bpp
```

This keeps the paper metric honest while allowing CoSER-DiC to test whether a
very small number of extra bits can close the large perceptual gap to official
CoD-Lite.

## Why This Is Needed

Current no-extra-bit Stage 4 is outside the official CoD-Lite perceptual curve:

```text
current CoSER Kodak LPIPS/DISTS:
  LPIPS best checked: about 0.4678
  DISTS best checked: about 0.3126

official CoD-Lite Kodak512 worst stored point:
  LPIPS: 0.3390
  DISTS: 0.2089
```

Adapter-only changes have been too small. Partial backbone finetuning is the
next no-extra-bit path, but a tiny counted control stream is also a legitimate
Stage 5 route:

```text
CoD-Lite sends native codec tokens.
CoSER can send semantic/detail tokens plus a much smaller condition residual
control signal.
```

The research question becomes:

```text
Can CoSER's semantic/detail stream plus 0.001-0.004 bpp of condition control
beat official CoD-Lite's perceptual/rate curve or give a better
rate-perception-faithfulness tradeoff?
```

## Current Implementation

Added:

```text
src/coserdic/entropy/control_grid.py
tests/test_control_grid.py
```

Codec:

```text
UniformControlGridCode
```

It uses fixed-bit uniform quantization and the existing fixed-bit payload
packer. This makes the payload byte count exact and auditable.

Evaluation support:

```text
scripts/eval_stage4_cod_lite_adapter.py

--counted-control-mode condition_residual_grid
--counted-control-mode condition_residual_dct
--counted-control-mode condition_residual_basis
--counted-control-mode condition_residual_affine
--control-grid-size
--control-groups
--control-dct-coeffs-per-group
--control-basis
--control-basis-components
--control-bits
--control-range
--control-scale
```

The evaluation path actually:

```text
1. computes an encoder-side target condition residual from the reference image
2. groups condition channels
3. spatially pools to a tiny grid
4. either transmits the grid directly or applies a fixed orthonormal DCT
5. for DCT mode, keeps fixed zigzag low-frequency coefficients
6. quantizes to fixed-bit codes
7. encodes to real bytes
8. decodes those bytes
9. expands the decoded grid/coefficient reconstruction back into condition space
10. adds it before CoD-Lite image decoding
11. adds control_payload_bpp to actual_payload_bpp
```

DCT mode sends no coefficient positions because the decoder knows the fixed
zigzag order:

```text
--counted-control-mode condition_residual_dct
```

This is the preferred first sweep because it can spend far fewer bits than a
full spatial grid while still correcting global condition structure.

Basis mode is the stronger follow-up:

```text
--counted-control-mode condition_residual_basis
```

It fits a fixed PCA/SVD basis from training-set condition residuals. The basis
is decoder-side fixed model state. Per image, only quantized coefficients are
transmitted and counted. This is more adaptive than DCT while keeping the same
bitstream accounting discipline.

Basis coefficients can now be transmitted with either:

```text
fixed_bits:
  uniform fixed-bit packing, exact constant bytes per image

huffman:
  position-conditioned static Huffman code over quantized basis coefficients
  fitted from non-eval train-cache coefficients
```

Basis coefficient selection can be:

```text
prefix:
  transmit the first K PCA/SVD coefficients

topk:
  search a fixed decoder-side candidate prefix C and transmit the image-specific
  top-K coefficient indices plus their quantized values
```

For `topk`, both the index stream and the value stream are image-specific
payload and are counted in `actual_payload_bpp`. This is a mainline Stage 5
information-flow improvement: it spends the tiny control budget on the basis
directions that matter for the current image rather than assuming the same
prefix is optimal for every image.

Sparse top-k can use:

```text
fixed_bits:
  fixed-bit index stream + fixed-bit value stream

huffman:
  position-conditioned static Huffman index stream
  + position-conditioned static Huffman value stream
  fitted from non-eval train-cache top-k selections
```

The sparse top-k Huffman prior is fixed decoder-side model state. The per-image
selected indices and values are still encoded into real byte streams and their
exact byte lengths are counted.

The scalar quantizer can be:

```text
uniform:
  linear quantization over [-range, range]

mu_law:
  deterministic mu-law companded quantization over [-range, range]
  spends more levels around zero for peaked condition-residual coefficients
```

The Huffman code lengths are fixed decoder-side model state. The transmitted
per-image payload is still the real encoded byte stream, and
`actual_payload_bpp` is computed from `len(payload)` for each image. This is a
direct BD-rate-oriented improvement over fixed-bit control when the coefficient
distribution is sharply peaked. The mu-law compander parameters are also fixed
decoder-side model state; they are not per-image side information.

Affine condition residual control is tracked separately in:

```text
docs/research/design_decisions/044_stage5_counted_affine_condition_control.md
```

It transmits group-wise condition residual gain/bias values:

```text
target_cond - base_cond ~= gain * (pred_cond - base_cond) + bias
```

This is counted as image-specific payload and targets adapter scale/bias errors
before CoD-Lite decoding.

The basis fit also records coefficient distribution statistics:

```text
coefficient_abs_mean
coefficient_abs_std
coefficient_abs_max
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

Use these values to choose `--control-range`. For the first fair sweep, prefer
range values near training p95 or p99, then verify clipping indirectly through
`control_condition_l1_delta` and final decoded metrics. Do not choose control
range purely by test-set feedback. Use `basis_reconstruction_stats` to avoid
spending GPU time on low-payload settings that explain very little condition
residual energy.

Added:

```text
scripts/sweep_stage5_counted_control.py
scripts/fit_stage5_condition_control_basis.py
scripts/inspect_stage5_control_basis.py
scripts/plan_stage5_control_curve.py
scripts/select_stage5_control_candidates.py
scripts/promote_stage5_control_candidates.py
scripts/collect_bd_curve_points.py
tests/test_sweep_stage5_counted_control.py
tests/test_fit_stage5_condition_control_basis.py
tests/test_inspect_stage5_control_basis.py
tests/test_plan_stage5_control_curve.py
tests/test_select_stage5_control_candidates.py
tests/test_promote_stage5_control_candidates.py
tests/test_collect_bd_curve_points.py
```

The sweep script can dry-run or execute multiple grid/DCT/basis settings and
records the expected summary paths plus exact control bpp. It can also read the
`recommended_basis_settings.json` emitted by the inspect script, which avoids
manual p95/p99 range transcription.

Example dry run:

```bash
.venv/bin/python scripts/sweep_stage5_counted_control.py \
  --checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_detailaware_adapter_bpp0312_transfer_ft600_b2.pt \
  --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl \
  --per-image-metrics results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/per_image_metrics.jsonl \
  --run-prefix 20260629_stage5_counted_control_bpp0312_limit64 \
  --dry-run \
  --output-json results/stage5_counted_control/20260629_bpp0312_limit64_sweep_plan.json
```

Fit a basis after GPU restart:

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

Inspect recommended basis ranges:

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

For BD-rate preparation, first select a multi-rate control curve plan from the
inspected settings:

```bash
.venv/bin/python scripts/plan_stage5_control_curve.py \
  --settings-json outputs/stage5_control_basis/20260629_stage5_bpp0312_condition_basis_g16_s8_k64_train8192/recommended_basis_settings.json \
  --band 0:0.00025:2 \
  --band 0.00025:0.001:2 \
  --band 0.001:0.002:2 \
  --rank-by retained_per_bpp \
  --min-basis-retained-energy 0.50 \
  --max-basis-residual-energy 0.50 \
  --max-quantization-rmse 0.20 \
  --max-clipped-fraction 0.10 \
  --include-selection prefix --include-selection topk \
  --include-codec fixed_bits --include-codec huffman \
  --include-quantizer uniform --include-quantizer mu_law \
  --output-json results/stage5_counted_control/20260629_bpp0312_curve_settings.json
```

Then dry-run a basis sweep directly from the curve settings:

```bash
.venv/bin/python scripts/sweep_stage5_counted_control.py \
  --checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_detailaware_adapter_bpp0312_transfer_ft600_b2.pt \
  --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl \
  --per-image-metrics results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/per_image_metrics.jsonl \
  --control-basis outputs/stage5_control_basis/20260629_stage5_bpp0312_condition_basis_g16_s8_k64_train8192/control_basis.pt \
  --run-prefix 20260629_stage5_basis_control_bpp0312_limit64 \
  --settings-json results/stage5_counted_control/20260629_bpp0312_curve_settings.json \
  --dry-run \
  --output-json results/stage5_counted_control/20260629_bpp0312_basis_limit64_sweep_plan.json
```

After executing a limit64 sweep, rank candidates before promoting to full552:

```bash
.venv/bin/python scripts/select_stage5_control_candidates.py \
  --sweep-plan results/stage5_counted_control/20260629_bpp0312_basis_limit64_sweep_plan.json \
  --reference-curve docs/research/baselines/cod_lite_kodak512_official_curve.json \
  --reference-metric lpips --reference-metric dists \
  --metric lpips --metric dists --metric psnr --metric ms_ssim \
  --min-basis-retained-energy 0.50 \
  --max-basis-residual-energy 0.50 \
  --max-psnr-drop 0.75 \
  --max-ms-ssim-drop 0.025 \
  --max-lpips-increase 0.0 \
  --max-dists-increase 0.0 \
  --top-k 8 \
  --output-json results/stage5_counted_control/20260629_bpp0312_basis_limit64_candidate_selection.json
```

This selector uses `actual_payload_bpp` only. It marks Pareto candidates and
filters obvious fidelity collapses before full552 promotion. When
`basis_reconstruction_stats` are present in the sweep plan, it also applies a
basis-explanation guard so low-payload settings that explain little
condition-residual energy are not included in `recommended`. It is not a
replacement for visual audit or final BD-rate curves.

When a reference curve is provided, it also reports:

```text
reference_lpips_at_bpp / reference_dists_at_bpp
lpips_gap_vs_reference_at_bpp / dists_gap_vs_reference_at_bpp
reference_bpp_for_lpips / reference_bpp_for_dists
*_single_point_rate_proxy_percent
```

These are same-bpp interpolation and single-point proxy diagnostics only. They
must not be reported as BD-rate. Their purpose is to prevent promoting
settings that still obviously trail the official CoD-Lite curve.

Generate full552 commands from guarded Pareto candidates:

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

The promotion script does not run evaluation. It only rewrites the original
limit64 eval commands, preserving the selected control settings while changing
the dataset limit and run name.

After running those full552 commands, collect the promoted summaries into a
curve JSON directly from the promotion JSON:

```bash
.venv/bin/python scripts/collect_bd_curve_points.py \
  --promotion-json results/stage5_counted_control/20260629_bpp0312_full552_promotion_commands.json \
  --name coserdic_stage5_bpp0312_control_curve \
  --dataset gencodec512_full552 \
  --stage stage4 \
  --output-json results/bd_curves/coserdic_stage5_bpp0312_control_curve.json
```

`collect_bd_curve_points.py` resolves each promoted command's
`--output-dir/--run-name/summary.json`, then records `actual_payload_bpp` as
the curve rate. It can still accept explicit `--input label=summary.json`
entries, but the promotion JSON path is preferred for promoted Stage 5 sweeps
because it avoids hand-copying paths.

If patch-FID/KID is computed in a separate JSON, merge it into the same curve
point with a matching label:

```bash
.venv/bin/python scripts/collect_bd_curve_points.py \
  --input candidate_a=results/stage4_cod_lite_adapter_eval/candidate_a/summary.json \
  --extra-metric-json candidate_a=results/analysis/image_distribution_metrics/candidate_a_patch_fid_kid.json \
  --metric lpips --metric dists --metric fid --metric kid \
  --name coserdic_stage5_candidate_a_kodak_curve \
  --dataset Kodak512 \
  --stage stage4 \
  --output-json results/bd_curves/coserdic_stage5_candidate_a_kodak_curve.json
```

Only after full552 summaries have been collected should BD-rate be computed:

```bash
.venv/bin/python scripts/compute_bd_rate.py \
  --reference docs/research/baselines/cod_lite_kodak512_official_curve.json \
  --candidate results/bd_curves/coserdic_stage5_bpp0312_control_curve.json \
  --metric lpips --metric dists \
  --output-json results/bd_curves/coser_stage5_vs_cod_lite_bdrate.json
```

`compute_bd_rate.py` records curve metadata and emits protocol warnings for
dataset or bpp-policy mismatches. Use `--fail-on-dataset-mismatch` for
protocol-matched promotion gates.

If the candidate curve was evaluated on full552 while the reference curve is
Kodak512, treat the result as an engineering diagnostic only. A paper claim
requires a protocol-matched reference and candidate curve.

The evaluation summary/per-image rows also report:

```text
stage3_actual_payload_bpp
control_payload_bpp
control_payload_bytes
pre_control_condition_l1
condition_l1
control_condition_l1_delta
```

Use these to separate "the control stream actually improves condition
recovery" from "the image metrics changed for another reason."

For basis mode, the evaluation summary additionally records:

```text
control_basis_source
control_basis_explained_variance
control_basis_cumulative_explained_variance
control_basis_coefficient_abs_quantiles
control_basis_coefficient_abs_mean/std/max
```

The decoder does not use the reference image. It only uses:

```text
decoded CoSER semantic/detail tensors
fixed model weights
decoded control payload bytes
```

## Bpp Examples at 512x512

For fixed-bit control:

```text
control_bytes = ceil(groups * grid_size * grid_size * bits / 8)
control_bpp   = 8 * control_bytes / (512 * 512)
```

Examples:

| groups | grid | bits | bytes | bpp |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 4x4 | 4 | 32 | 0.000977 |
| 8 | 4x4 | 4 | 64 | 0.001953 |
| 16 | 4x4 | 4 | 128 | 0.003906 |
| 8 | 8x8 | 4 | 256 | 0.007812 |

Grid mode is useful as a stronger but less bit-efficient reference.

DCT examples:

| groups | grid | coeffs/group | bits | bytes | bpp |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 8x8 | 2 | 4 | 8 | 0.000244 |
| 8 | 8x8 | 4 | 4 | 16 | 0.000488 |
| 16 | 8x8 | 4 | 4 | 32 | 0.000977 |
| 16 | 8x8 | 8 | 4 | 64 | 0.001953 |

Basis examples:

| coeffs | bits | bytes | bpp |
| ---: | ---: | ---: | ---: |
| 8 | 4 | 4 | 0.000122 |
| 16 | 4 | 8 | 0.000244 |
| 32 | 4 | 16 | 0.000488 |
| 64 | 4 | 32 | 0.000977 |

Huffman basis control has image-dependent payload length. The inspect/sweep
planning tools report train-cache mean payload bytes for the requested
coefficient prefix, while actual eval rows report the exact per-image bytes and
bpp. Use the Huffman setting only when the same basis checkpoint contains the
matching `control_huffman_priors` key, such as `p95_b4` or `p99_b4`.
For mu-law priors, keys include the quantizer label, such as `p95_b4_mulaw16`.

The first Stage 5 control sweep should start with basis-prefix, sparse
basis-topk, and DCT in the 0.00012-0.002 bpp range, then compare against grid
control in the 0.001-0.004 bpp range. For sparse top-k, the index stream raises
the payload, so candidate prefix C should be filtered by planned
`control_bpp` before running full552.

## GPU-Restart Sweep

GPU is currently unavailable:

```text
nvidia-smi: Failed to initialize NVML: Unknown Error
torch.cuda.is_available: False
```

After container restart, evaluate a conservative control sweep on limit64
before full552. Use the current strongest bpp0.0312 or partial-backbone
checkpoint if available.

Example with the existing bpp0.0312 transfer anchor:

```bash
.venv/bin/python scripts/eval_stage4_cod_lite_adapter.py \
  --checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_detailaware_adapter_bpp0312_transfer_ft600_b2.pt \
  --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl \
  --per-image-metrics results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/per_image_metrics.jsonl \
  --run-name 20260629_stage5_counted_control_bpp0312_g8_grid4_b4_limit64_eval \
  --crop-size 512 --limit 64 --batch-size 4 --num-workers 4 \
  --blend-alpha 1.0 \
  --counted-control-mode condition_residual_grid \
  --control-groups 8 --control-grid-size 4 --control-bits 4 \
  --control-range 0.25 --control-scale 1.0 \
  --wandb-mode offline
```

Then test:

```text
mode=dct,  groups=8,  grid=8, coeffs=2, bits=4  (~0.00024 bpp)
mode=dct,  groups=8,  grid=8, coeffs=4, bits=4  (~0.00049 bpp)
mode=dct,  groups=16, grid=8, coeffs=4, bits=4  (~0.00098 bpp)
mode=dct,  groups=16, grid=8, coeffs=8, bits=4  (~0.00195 bpp)
mode=basis, groups=16, grid=8, coeffs=8,  bits=4  (~0.00012 bpp)
mode=basis, groups=16, grid=8, coeffs=8,  bits=4, codec=huffman  (train-mean bpp from basis prior)
mode=basis, groups=16, grid=8, coeffs=8,  bits=4, quantizer=mu_law
mode=basis, groups=16, grid=8, coeffs=8,  bits=4, codec=huffman, quantizer=mu_law
mode=basis, groups=16, grid=8, coeffs=16, bits=4  (~0.00024 bpp)
mode=basis, groups=16, grid=8, coeffs=16, bits=4, codec=huffman  (train-mean bpp from basis prior)
mode=basis, groups=16, grid=8, coeffs=16, bits=4, quantizer=mu_law
mode=basis, groups=16, grid=8, coeffs=16, bits=4, codec=huffman, quantizer=mu_law
mode=basis, groups=16, grid=8, coeffs=32, bits=4  (~0.00049 bpp)
mode=basis, groups=16, grid=8, coeffs=32, bits=4, codec=huffman  (train-mean bpp from basis prior)
mode=basis, groups=16, grid=8, coeffs=32, bits=4, quantizer=mu_law
mode=basis, groups=16, grid=8, coeffs=32, bits=4, codec=huffman, quantizer=mu_law
mode=basis, groups=16, grid=8, coeffs=64, bits=4  (~0.00098 bpp)
groups=4,  grid=4, bits=4  (~0.0010 bpp)
groups=8,  grid=4, bits=4  (~0.0020 bpp)
groups=16, grid=4, bits=4  (~0.0039 bpp)
groups=8,  grid=8, bits=3  (~0.0059 bpp)
groups=8,  grid=8, bits=4  (~0.0078 bpp)
```

Promotion to full552 requires:

```text
actual_payload_bpp includes control_payload_bpp
LPIPS/DISTS improve enough to move toward official CoD-Lite curve overlap
PSNR/MS-SSIM/structure do not collapse
visual samples show faithful structure
no RGB output blend is used as the method
```

## Interpretation Rules

This is valid to report only as:

```text
CoSER semantic/detail payload + counted condition-control stream
```

Do not compare it to no-extra-bit CoSER points without showing the added bpp.
Do not hide the control stream in model metadata.

If the control stream gives a large quality jump, the next research step is not
to overfit this hand-designed grid. Instead:

```text
train a learned control encoder/decoder
add entropy modeling for the control symbols
jointly allocate bits among semantic, detail, and control streams
build multi-rate curves and compute BD-rate
```

This path is compatible with partial CoD-Lite backbone adaptation:

```text
no-extra-bit partial backbone finetune first
tiny counted control stream if deterministic control saturates
then joint Stage 5 curve construction
```
