# Stage 5 Base-Fixed, Fixed-Target, and Hybrid Selector Probe

Date: 2026-06-29 JST

## Question

Can the current Stage 5 mainline close more of the CoD-Lite gap by changing
the counted-control training target or by using an encoder-side selector over
condition-control families?

This probe stays on the Core MVP mainline:

```text
CoSER semantic/detail actual bitstream
+ official CoD-Lite 0.0156 backbone
+ CoSER-owned adapter / LoRA / counted condition control
```

It does not transmit CoD-Lite native bitstreams and does not use RGB
postprocessing as the proposed method.

## Implementation Updates

`scripts/eval_stage4_cod_lite_adapter.py` and
`scripts/train_stage4_cod_lite_adapter.py` now support:

```text
condition_base_affine_basis
```

This lets the encoder send a fixed teacher/base condition residual while the
adapter learns the complementary condition.

The training loop was also corrected for counted-control runs:

```text
adapter_target_cond = target_cond - decoded_control_correction
```

The detail auxiliary losses now compare the raw adapter condition to this
adapter target rather than to the post-control teacher condition. This prevents
the detail branch from trying to explain information already carried by the
counted control stream.

## Base-Fixed Control Result

Base-fixed control was tested to see whether a stable, base-relative control
stream is easier for the adapter to complement than a moving post-adapter
residual.

Limit64 results:

| run | bpp | PSNR | MS-SSIM | LPIPS | DISTS | control delta L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| basefixed old basis | 0.015239 | 21.6299 | 0.7389 | 0.3794 | 0.2713 | +0.0027 |
| basefixed new basis | 0.015111 | 21.6166 | 0.7388 | 0.3779 | 0.2705 | +0.0041 |
| basefixed prefix32+topk32 | 0.016011 | 21.6157 | 0.7388 | 0.3772 | 0.2700 | +0.0039 |
| basefixed complement FT500 | 0.015111 | 21.7000 | 0.7400 | 0.3818 | 0.2727 | -0.0056 |
| basefixed complement FT500 prefix32+topk32 | 0.016011 | 21.7008 | 0.7400 | 0.3811 | 0.2720 | -0.0059 |

Interpretation:

```text
Base-fixed control can improve condition L1 after complement training, but it
pushes the model toward fidelity and away from LPIPS/DISTS. Increasing the
control basis information with prefix32+topk32 does not change that conclusion.
```

Decision:

```text
Do not promote the base-fixed branch.
Keep the implementation because it is useful for future complementary-control
experiments, but do not spend full552/FID runs on this exact recipe.
```

## Fixed-Target Control-Aware Training

The corrected counted-control training target was retested on the current
LoRA-all 0.0156 + k16 control mainline.

Full552 results:

| run | bpp | PSNR | MS-SSIM | LPIPS | DISTS | condition L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| old k16 anchor | 0.014978 | 21.1966 | 0.7117 | 0.3665 | 0.2575 | 0.4030 |
| fixedtarget FT600 | 0.014978 | 21.2643 | 0.7133 | 0.3649 | 0.2597 | 0.4014 |
| fixedtarget DISTS FT500 | 0.014977 | 21.2877 | 0.7151 | 0.3651 | 0.2610 | 0.4017 |

Interpretation:

```text
The corrected target is beneficial for fidelity and LPIPS, but it does not
improve DISTS. A direct DISTS-weighted continuation worsens DISTS further while
continuing to recover PSNR/MS-SSIM.
```

Decision:

```text
Do not promote either fixedtarget checkpoint as the new perceptual Stage 5
anchor. The fixedtarget FT600 checkpoint is a useful balanced/fidelity
diagnostic, but the old k16 anchor remains stronger on DISTS.
```

## Hybrid Selector Diagnostic

A guarded image-RDO selector was tested as an upper-bound diagnostic. The
encoder may choose a counted control family per image; the decoder receives the
mode selector plus the selected entropy-coded control payload.

Limit64 comparison:

| run | bpp | PSNR | MS-SSIM | LPIPS | DISTS | mode summary |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| k16 affine_basis | 0.014631 | 21.5865 | 0.7344 | 0.3526 | 0.2552 | fixed affine_basis |
| DISTS-RDO, no rate penalty | 0.035549 | 21.6047 | 0.7346 | 0.3507 | 0.2527 | mostly affine_grid |
| DISTS-RDO, rate lambda 0.5 | 0.014955 | 21.5853 | 0.7342 | 0.3538 | 0.2544 | affine_basis 45, affine 9, affine_dct 4, none 6 |
| LPIPS-RDO, rate lambda 0.5 | 0.015924 | 21.5932 | 0.7345 | 0.3519 | 0.2551 | affine_basis 53, affine_dct 11 |

Interpretation:

```text
The unpenalized selector proves that more condition-control bits can improve
DISTS, but it spends far too much rate. With rate penalty, the selector gives
only tiny metric-specific gains and no clear Pareto improvement.
```

Decision:

```text
Do not promote hybrid selector mode as a Stage 5 result.
Use it as an upper-bound diagnostic only. The main bottleneck is still the
quality of the control/detail representation, not mode selection among the
current control families.
```

## Current Mainline Decision

The active Stage 5 continuation remains:

```text
CoD_Lite_bpp_0_0156 denoiser-all LoRA
+ k16 sparse counted condition control
+ future work on stronger CoSER detail/control representation
```

The next mainline step should not be more fixed alpha, gate-only, base-fixed,
or selector-only tuning. It should change the information flow:

```text
1. make the transmitted detail residual more diffusion-control-friendly
2. add a trainable detail-control projection/head from the same counted detail payload
3. improve semantic/detail rate allocation for Stage 4 output, not only Stage 3 x_aux
4. then revisit counted-control or tiny control streams after the condition/detail representation improves
```

External-baseline status is unchanged:

```text
CoSER-DiC is not yet beating official CoD-Lite / CoD / RDVQ / GLC curves.
Current progress is internal and diagnostic, not a publishable external win.
```
