# Stage 4 DISTS-Loss Probe and Train-Cache Scale Decision

Date: 2026-06-29

## Decision

Keep the new DISTS training-loss hook, but do not promote the first
DISTS-guarded continuation as a Stage 4 anchor.

The next mainline move is larger clean train-cache scaling plus stronger
condition recovery, not heavier hand-tuning of image-space perceptual losses on
the 2048-image cache.

## What Changed

`scripts/train_stage4_cod_lite_adapter.py` now supports:

```text
--dists-weight
```

When enabled, the adapter training loop adds DISTS loss between the decoded
Stage 4 output and the reference image. This is a legal training-only
supervision signal: at inference time, the decoder still uses only decoded
CoSER semantic/detail payload, fixed adapter weights, and fixed CoD-Lite
backbone weights.

The payload policy is unchanged:

```text
paper_bpp = Stage 3 semantic/detail actual_payload_bpp
no image-specific DISTS target, prompt, map, gate, or reference feature is
transmitted
```

## Probe Run

Run:

```text
20260629_stage4_detailaware_ft600_distsguard_ft1000_b2
```

Init:

```text
checkpoints/stage4_cod_lite_adapter/
20260628_stage4_detailaware_adapter_idstart_ft600_b2.pt
```

Training data:

```text
20260628_stage3_gencodec512_traincache_clic_oi_2048_seed20260628
```

Main loss change:

```text
dists_weight = 0.12
stage3_l1_guard_weight = 1.00
stage3_mse_guard_weight = 2.00
```

Training summary:

| metric | value |
| --- | ---: |
| condition_l1_mean | 0.378976 |
| base_condition_l1_mean | 0.517867 |
| condition_l1_delta_vs_base_mean | -0.138891 |
| lpips_mean | 0.388660 |
| dists_loss_mean | 0.268097 |
| stage4_psnr_mean | 21.3806 |

## Limit64 Evaluation

Same first 64 images from the strict 512 full552 feature cache.

```text
actual_payload_bpp = 0.013654
```

| run | PSNR | MS-SSIM | LPIPS | DISTS | condition_l1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage 3 anchor | 22.3652 | 0.7568 | 0.5450 | 0.3497 | n/a |
| bpp0.0156 ft600 | 21.6722 | 0.7392 | 0.4142 | 0.2986 | 0.4170 |
| bpp0.0156 ft3000 | 21.7478 | 0.7418 | 0.4151 | 0.2992 | 0.4184 |
| bpp0.0312 transfer | 21.8379 | 0.7442 | 0.4215 | 0.2888 | n/a |
| ft600 + DISTS guard ft1000 | 21.8067 | 0.7420 | 0.4195 | 0.3035 | 0.4185 |

Split check for the DISTS-guarded run:

| split | count | PSNR | MS-SSIM | LPIPS | DISTS |
| --- | ---: | ---: | ---: | ---: | ---: |
| CLIC2020 test subset | 40 | 22.1951 | 0.7653 | 0.3831 | 0.2902 |
| Kodak | 24 | 21.1594 | 0.7032 | 0.4802 | 0.3256 |

## Judgment

The DISTS hook works and can be useful in later guarded training, but this
particular setting is not promoted:

```text
improves over Stage 3 in LPIPS/DISTS at the same payload
improves fidelity versus ft600 on the same limit64 subset
does not beat ft600 on LPIPS or DISTS
does not beat bpp0.0312 transfer on DISTS or fidelity
does not change the information flow enough to close the official CoD-Lite gap
```

This result reinforces the current mainline diagnosis:

```text
the bottleneck is condition recovery and diffusion control quality,
not merely the absence of one more perceptual image loss
```

## Next Mainline Action

Move from the 2048-image train-cache to a larger clean cache before drawing
ceiling conclusions about the adapter:

```text
short-term target:
  8192 strict 512 crops from CLIC train + OpenImages train

medium target:
  20k to 100k clean non-eval crops if adapter scaling is positive
```

Evaluation splits must remain untouched:

```text
Kodak24
CLIC2020 test428
DIV2K val100
```

The next adapter run should prioritize:

```text
condition imitation first
condition statistics / high-frequency matching
decoded semantic latent + decoded detail context
fidelity guard
optional modest LPIPS/DISTS terms
```

Avoid interpreting larger DISTS weight or fixed RGB blending as the method.
The Stage 4 method remains a CoSER-owned condition adapter over decoded
semantic/detail actual-bitstream features.
