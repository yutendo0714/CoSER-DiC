# Stage 5 Zero-Centered Detail Quantizer Probe

Date: 2026-06-30 JST

## Question

Can a low-bit detail residual stream be made more useful for Stage 4 / Stage 5
diffusion conditioning by fixing the residual quantizer itself, rather than by
adding another output-side adapter or postprocess?

The immediate suspicion was that the 2-bit residual grid path was unfairly weak:
with the old uniform quantizer, zero residual was not represented as an exact
zero code. For residual coding this is a serious mismatch because most cells
are close to zero.

## Implementation

Added a codec-level detail residual quantizer option:

```text
--detail-quantizer uniform
--detail-quantizer zero_centered
```

`zero_centered` gives the middle code an exact zero residual and maps negative
and positive sides separately. The implementation is carried through the static
residual Huffman priors, Stage 3 bitstream evaluation, fast train-cache
generation, and Stage 4 detail-code normalization.

Touched main files:

```text
src/coserdic/entropy/residual_grid.py
src/coserdic/entropy/__init__.py
scripts/fit_stage3_residual_huffman_prior.py
scripts/eval_stage3_uniform_residual_bitstream.py
scripts/build_stage3_training_cache_fast.py
scripts/train_stage4_cod_lite_adapter.py
scripts/eval_stage4_cod_lite_adapter.py
tests/test_residual_grid.py
```

Verification:

```text
.venv/bin/python -m py_compile \
  src/coserdic/entropy/residual_grid.py \
  scripts/fit_stage3_residual_huffman_prior.py \
  scripts/eval_stage3_uniform_residual_bitstream.py \
  scripts/build_stage3_training_cache_fast.py \
  scripts/train_stage4_cod_lite_adapter.py \
  scripts/eval_stage4_cod_lite_adapter.py

.venv/bin/python -m pytest tests/test_residual_grid.py -q
10 passed
```

Also verified the Stage 4 detail-code normalization path directly:

```text
zero_centered 2-bit codes [0, 1, 2, 3] -> [-1.0, -0.5, 0.0, 1.0]
```

## Stage 3 Prior

Fitted prior:

```text
20260629_stage3_residual_semposleft_g4_sm0_d16_b2_zc_r025_crop512_2048calib_stage1_rateprior0005_b16
```

Prior summary:

| item | value |
| --- | ---: |
| detail bits | 2 |
| downsample factor | 16 |
| quantizer | zero_centered |
| counts | [8716, 280761, 5929493, 72486] |
| symbol entropy | 0.3681 bits |
| mean Huffman bits/symbol | 1.0678 |
| clip ratio | 0.000812 |

The exact-zero symbol dominates the code distribution. This confirms the old
uniform 2-bit detail path was structurally mismatched to residual coding.

## Stage 3 Limit64 Screen

Run:

```text
20260629_stage3_d16_b2_zc_topk2048_gencodec512_limit64_feature_cache
```

Strict decoded-bitstream results:

| setting | bpp | detail bpp | PSNR | MS-SSIM | LPIPS | DISTS | detail entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| d16 b2 uniform | 0.024131 | 0.015335 | 20.8772 | 0.7002 | 0.6952 | 0.3708 | 1.0038 |
| d16 b2 zero-centered | 0.021151 | 0.012354 | 22.2556 | 0.7588 | 0.5496 | 0.3464 | 0.2697 |
| d16 b3 uniform | 0.025829 | 0.017033 | 22.4673 | 0.7642 | 0.5945 | 0.3508 | 1.3219 |

Interpretation:

```text
zero-centered d16/b2 fixes the low-bit residual failure.

It is much better than uniform d16/b2, and it gives better LPIPS/DISTS than
uniform d16/b3 at lower actual payload bpp. d16/b3 still has slightly better
PSNR/MS-SSIM.
```

## Stage 4 / Stage 5 Limit64 Screen

First screened the d16/b2 zero-centered Stage 3 cache with the existing d32
detail-control adapter, then trained a dedicated short d16/b2 zero-centered
adapter continuation from the d16/b3 checkpoint.

Runs:

```text
20260629_stage5_detailcontrol_fusion_on_stage3_d16_b2_zc_fixednorm_limit64_eval
20260629_stage5_d16b2zc_detailcontrol_fusion_fromd16b3_ft400_limit64_eval
```

Comparison:

| setting | bpp | control bpp | PSNR | MS-SSIM | LPIPS | DISTS | condition L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| d32 detail-control fusion | 0.014632 | 0.000978 | 21.6464 | 0.7365 | 0.3512 | 0.2577 | 0.4011 |
| d16 b2 zc + d32 adapter | 0.022122 | 0.000972 | 21.2761 | 0.7339 | 0.3622 | 0.2523 | 0.4129 |
| d16 b2 zc dedicated FT400 | 0.022120 | 0.000970 | 21.5267 | 0.7361 | 0.3523 | 0.2543 | 0.4064 |
| d16 b3 dedicated FT400 | 0.026810 | 0.000981 | 21.7973 | 0.7439 | 0.3499 | 0.2499 | 0.4090 |

Interpretation:

```text
d16/b2 zero-centered transfers to Stage 4 much better than old d16/b2 uniform.
The dedicated adapter recovers fidelity and nearly matches the d32 LPIPS anchor,
while improving DISTS slightly.

But it spends much more actual payload bpp than the d32 anchor and still does
not produce an external-baseline curve win. It is not a Stage 5 promotion.
```

## Decision

Keep the zero-centered residual quantizer as mainline codec infrastructure.

Do not promote the d16/b2 zero-centered checkpoint as a Stage 5 result.

The useful conclusion is architectural:

```text
detail residual quantization and entropy design matter more than another
adapter-only tweak.
```

Next mainline moves:

```text
1. test zero-centered / dead-zone detail quantization at d32 and other
   semantic/detail allocation points
2. add a semantic-conditioned learned residual entropy model so detail bpp can
   drop without weakening the Stage 4 condition signal
3. train Stage4-aware detail heads from the same transmitted residual payload
4. only promote candidates after full552 and curve-level comparison against
   official CoD-Lite / CoD-style baselines
```

This keeps the research centered on the CoSER-DiC bitstream and decoder
conditioning mechanism, not on posthoc RGB manipulation.
