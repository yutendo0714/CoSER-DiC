# 062 Mainline Diagnosis and Next Research Policy

Date: 2026-06-29 JST

## Decision

Adopt the current external advice as the active CoSER-DiC mainline research
policy.

The main diagnosis is:

```text
CoSER-DiC is making real internal progress, but it is not yet externally
competitive against official CoD-Lite / CoD / RDVQ-style baselines.

The bottleneck is not simply that CoD-Lite is strong.
The bottleneck is that decoded CoSER semantic/detail features are not yet
converted into the condition representation that a pretrained compression
diffusion backbone can use with both high fidelity and high realism.
```

Therefore, do not optimize for small visible improvements or post-hoc output
changes. Optimize the core information path:

```text
entropy-coded semantic/detail payload
-> decoded CoSER features
-> condition recovery
-> pretrained diffusion backbone
-> fidelity-preserving perceptual reconstruction
```

This preserves the Core MVP: CoSER semantic/detail streams remain the codec
payload, optional image-specific control streams are counted in
`actual_payload_bpp`, and CoD / CoD-Lite are pretrained decoder backbones or
baselines rather than replacements for the CoSER codec.

## Current Position

The strict 512 GenCodec-style evaluation split remains:

```text
Kodak24 + CLIC2020 test428 + DIV2K val100 = full552
crop 512 x 512
main bpp: actual_payload_bpp / paper_bpp
```

Stage 1-3 is a stable actual-bitstream substrate:

```text
actual_payload_bpp 0.013999
semantic/detail bpp about 0.008997 / 0.005002
roundtrip semantic/detail true
```

It is not a competitive final codec by itself, but it is valid for testing the
diffusion-backbone hypothesis.

The current no-extra-bit Stage 4 anchor is:

```text
20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8_full552_eval
actual_payload_bpp 0.013999
PSNR     21.2351
MS-SSIM   0.7166
LPIPS     0.4139
DISTS     0.2821
condition_l1 0.4108
```

The current no-extra-bit balanced condition gate is:

```text
20260629_stage4_condition_strength_gate_balanced_spatial_ft800_b2ga2_full552_eval
actual_payload_bpp 0.013999
PSNR     21.2121
MS-SSIM   0.7154
LPIPS     0.4068
DISTS     0.2786
condition_l1 0.4106
```

The current counted-control anchors are:

```text
balanced sparse counted control:
  actual_payload_bpp 0.015486
  control_payload_bpp about 0.00149
  PSNR 21.1617 / MS-SSIM 0.7149 / LPIPS 0.3891 / DISTS 0.2691

fixed perceptual control_scale=1.25:
  actual_payload_bpp 0.015488
  control_payload_bpp about 0.00149
  PSNR 21.1319 / MS-SSIM 0.7140 / LPIPS 0.3844 / DISTS 0.2663

post-control reused balanced gate:
  actual_payload_bpp 0.015486
  control_payload_bpp about 0.00149
  PSNR 21.1249 / MS-SSIM 0.7129 / LPIPS 0.3813 / DISTS 0.2650

true control-aware perceptual gate:
  actual_payload_bpp 0.015486
  control_payload_bpp about 0.00149
  PSNR 21.1187 / MS-SSIM 0.7126 / LPIPS 0.3780 / DISTS 0.2635
```

The true control-aware perceptual gate improves LPIPS/DISTS at the same counted
payload but loses fidelity. It is a perceptual operating point, not a balanced
Stage 5 promotion.

Its GenCodec-style patch-FID/KID is mixed:

```text
Kodak24:
  FID 78.8185 / KID 0.021144

CLIC2020 test428:
  FID 57.1402 / KID 0.006855

DIV2K val100:
  FID 153.0634 / KID 0.013843
```

Relative to the previous post-control reused balanced-gate probe, it improves
CLIC distribution metrics but slightly regresses Kodak and DIV2K. Therefore,
the next gate/control work must include distribution or structure guards
instead of optimizing LPIPS/DISTS alone.

Official CoD-Lite Kodak512 at nominal 0.015625 bpp remains much stronger:

```text
PSNR     20.7667
MS-SSIM   0.7090
LPIPS     0.2259
DISTS     0.1402
FID      38.4785
```

Do not claim external-baseline superiority.

## What Counts as Mainline

A change is mainline if it changes at least one of:

```text
1. the information transmitted by semantic/detail streams,
2. entropy coding or rate allocation,
3. how decoded CoSER features condition the pretrained diffusion backbone,
4. content-dependent gate/control over diffusion strength,
5. actual_payload_bpp curves rather than one operating point.
```

A change is not mainline if it only changes RGB output appearance after
decoding without improving information flow or decoder conditioning.

Keep these only as diagnostics or bracketing tools:

```text
fixed alpha blend
RGB output blending
ResUNet / CNN refiner
hand-tuned control scale
LPIPS/DISTS-only loss fiddling
single-subset wins
condition_l1-only promotion
single-bpp claims
```

## Active Bottleneck

The active bottleneck is condition recovery and diffusion control:

```text
condition_l1 improves, but final image metrics do not always improve;
predicted conditions are still not faithful enough;
stronger prior use improves LPIPS/DISTS/FID but damages PSNR/MS-SSIM;
detail context is still weaker than it should be for diffusion control;
counted control helps distribution, but costs bits and has not closed the
official CoD-Lite gap.
```

This means the next improvements must be structural, not cosmetic.

The first full552 condition-statistics audit for the current Stage 4 anchor is:

```text
run:
  20260629_stage4_detailcontrast_condition_stats_full552

summary:
  results/stage4_cod_lite_condition_stats/20260629_stage4_detailcontrast_condition_stats_full552/summary.json

base -> target:
  L1            0.5372
  relative L2   0.8761
  cosine        0.6220

pred -> target:
  L1            0.4108
  relative L2   0.6696
  cosine        0.7573

win rates:
  pred L1 better than base        99.64%
  pred relative L2 better than base 99.82%
```

This confirms that the adapter is not noise: decoded CoSER semantic/detail
features move the CoD-Lite condition strongly toward the native reference
condition.

However, the same audit shows why condition L1 alone is not sufficient:

```text
target condition:
  std mean                  0.7953
  spatial highfreq ratio    0.2575

base condition from Stage3 x_aux:
  std mean                  0.8235
  spatial highfreq ratio    0.2776

predicted condition:
  std mean                  0.7417
  spatial highfreq ratio    0.2400

pred - target:
  std delta                -0.0536
  mean delta               -0.0054
```

Interpretation:

```text
The current adapter improves alignment but produces a condition that is too
low-energy and too smooth compared with the native CoD-Lite condition. This is
consistent with the observed tradeoff: stronger diffusion prior use improves
LPIPS/DISTS but damages fidelity, while the current condition recovery cannot
yet deliver both.
```

Therefore the next adapter objective should explicitly include channel
mean/std matching and spatial spectrum/high-frequency matching, with image
fidelity guards. Do not keep optimizing only `condition_l1`.

## Priority Order

### 1. Condition-Imitation-First Adapter

Use official CoD-Lite native condition as a training teacher:

```text
training target:
  target_cond = official CoD-Lite native condition from reference image

inference:
  pred_cond = adapter(decoded CoSER tensors)
```

No target condition or original image information is transmitted at inference.

Recommended curriculum:

```text
phase A:
  condition L1 + cosine

phase B:
  channel mean/std matching
  spatial spectrum / high-frequency condition matching

phase C:
  image L1 + LPIPS + DISTS
  Stage3 anchor/fidelity guard

phase D:
  deterministic gate/control losses
```

Promotion must use final decoded metrics, not condition L1 alone.

### 2. Real Condition-Space Control Gate

The gate should operate in condition space, not RGB output space.

Inputs should include decoded CoSER features:

```text
x_aux
x_sem
residual magnitude
semantic latent statistics
detail code/confidence features
adapter delta norm
base condition quality proxy
optional counted-control confidence
```

Outputs may be:

```text
global strength
spatial strength
channel/spatial condition gate
region-wise control strength
```

Success requires beating the current no-extra balanced gate and counted-control
operating points at the same `actual_payload_bpp`, without large fidelity loss.

### 3. Diffusion-Friendly Detail Representation

Current detail residual is still too reconstruction-oriented.

Add a Stage4-aware detail-control head from the same transmitted detail payload:

```text
same detail payload ->
  reconstruction detail feature for x_aux
  diffusion-control detail feature for adapter/gate/control
```

Do not add bits first. Freeze the transmitted stream, train the extra
diffusion-control projection/head, then decide whether to unfreeze the detail
decoder or encoder with actual entropy checks.

Success requires detail-zero and detail-shuffle ablations to hurt more than
they do now.

### 4. Stage2/Stage3 Rate Allocation and Entropy

The Stage3-optimal semantic/detail split may not be Stage4/5-optimal.

Build actual-payload operating points around:

```text
0.008, 0.010, 0.012, 0.014, 0.016, 0.020, 0.031 bpp
```

Sweep:

```text
more semantic / less detail
less semantic / more detail
higher quality semantic / lower detail
lower semantic / stronger diffusion detail
small counted control versus more detail bits
```

Improve entropy coding where it moves curves:

```text
semantic learned categorical/context prior
RDVQ-inspired top-k/escape ANS
semantic-conditioned residual Gaussian conditional
hyperprior or small side stream if counted
learned quantization scale
```

### 5. Matched Baseline Curves

No BD-rate or external win claim from one point.

Required curves:

```text
CoSER no-extra adapter curve
CoSER balanced gate curve
CoSER counted-control curve
official CoD-Lite actual or clearly labeled nominal/payload curve
CoD / one-step CoD if reproducible
RDVQ real-bitstream curve if feasible
GLC / StableCodec only with clear bpp labeling
```

Metrics:

```text
PSNR / MS-SSIM guards
LPIPS / DISTS
GenCodec-style patch-FID/KID
edge consistency
DINO similarity
segmentation/OCR/small-object subsets if available
```

## Immediate Execution Policy

For the next research block:

```text
1. Freeze current anchors and do not re-label them as external wins.
2. Evaluate any new gate/control on full552 and split summaries.
3. Add mandatory semantic/detail/control ablations before promotion.
4. Log native/reference/base/predicted/controlled condition statistics.
5. Start condition-imitation-first adapter training before more hand-tuned
   scale sweeps.
6. Start diffusion-friendly detail-control head from the same detail payload.
7. Build the first actual_payload_bpp curve.
```

## Success Criteria

Short-term:

```text
Stage4/5 beats Stage3 at same or counted actual_payload_bpp;
improvements hold across Kodak / CLIC2020 / DIV2K;
semantic/detail ablations prove CoSER bitstream information is used;
no hidden reference leakage;
no bpp accounting ambiguity.
```

Medium-term:

```text
LPIPS/DISTS/FID gap to official CoD-Lite is substantially reduced;
PSNR/MS-SSIM and structure do not collapse;
counted-control curve gives a better tradeoff than no-extra curve;
actual BD-rate curves exist.
```

Long-term:

```text
CoSER wins on LPIPS/DISTS BD-rate, or on a clearly measured
rate-perception-faithfulness tradeoff, against matched CoD-Lite / CoD / RDVQ
protocols.
```

## Final Guardrail

Do not let the project drift back into fixed alpha, RGB postprocessing, or
single-point metric tuning.

The idea is not dead. The current evidence says the bottleneck is specific:

```text
condition recovery and diffusion control are still weak.
```

The next best path is to improve that information path directly.
