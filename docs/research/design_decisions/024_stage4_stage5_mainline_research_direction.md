# Stage 4 / Stage 5 Mainline Research Direction

Date: 2026-06-28 JST  
Status: Active mainline policy

## High-Level Judgment

Current CoSER-DiC has meaningful internal progress, but it is not yet a
publishable external-baseline win.

Current interpretation:

```text
stable semantic/detail actual-bitstream substrate
working Stage 4 CoD-Lite adapter path
decoded CoSER features can improve Stage 3 without extra payload
no demonstrated superiority yet over official CoD-Lite, CoD, RDVQ, GLC, or StableCodec
```

The central method remains:

```text
semantic VQ token stream
+ detail residual latent stream
+ entropy-coded actual payload
+ compression-oriented pretrained diffusion backbone
+ CoSER-owned conditioning adapter
```

Fixed alpha blends are diagnostics only. They should not become the method.
The next work must improve information flow, condition recovery, gating, rate
allocation, and baseline comparisons.

## Current Anchors

Strict 512 Stage 3 anchor:

```text
dataset:
  Kodak24 + CLIC2020 test428 + DIV2K val100
  total: 552

actual_payload_bpp:
  0.013999

semantic/detail bpp:
  about 0.009 / 0.005

roundtrip:
  semantic=true
  detail=true
  failure_count=0

metrics:
  PSNR about 21.99
  MS-SSIM about 0.7348
  LPIPS about 0.576
  DISTS about 0.354
```

Stage 4 current state:

```text
CoD-Lite wrapper loads official checkpoints
frozen CoD-Lite backbone decodes from CoSER-produced condition tensors
decoded 256-channel semantic latent is better than RGB semantic image input
decoded detail context improves condition prediction
fixed alpha0.25/0.30 are stability anchors, not paper operating points
raw adapter output improves perceptual metrics but damages fidelity
```

Official CoD-Lite remains far ahead in Kodak512 perceptual metrics at comparable
rate. Do not claim external-baseline superiority or Stage 5.

## What Is Off-Mainline

Do not optimize these as the main method:

```text
ResUNet decoder refiner
fixed alpha blend hand-tuning
cosmetic postprocess-only improvements
single-point claims without curves
condition_l1-only promotion
reference-derived decode-time choices
debug_full_stream_bpp paper reporting
```

Fixed blends may remain useful as probes because they are deterministic
decoder-side settings and do not add payload. They do not solve condition
recovery, content-dependent diffusion use, or semantic/detail control.

## Priority 1: Real CoSER-to-CoD-Lite Condition Adapter

Main adapter inputs should come from decoded CoSER state:

```text
Stage 3 decoded x_aux
semantic-only reconstruction x_sem
residual_hat = x_aux - x_sem
decoded semantic latent, 256ch
decoded residual_grid_hat
normalized detail_codes
optional fixed or counted rate/perception condition
```

Training may use official CoD-Lite native condition from the reference image as
a teacher. Inference must use only:

```text
pred_cond = adapter(decoded CoSER tensors)
```

Preferred condition form:

```text
base_cond = CoD-Lite native condition from decoded Stage3 x_aux
pred_cond = base_cond + scale * tanh(adapter_delta)
```

Use condition imitation as the first phase:

```text
condition L1
condition cosine loss
channel mean/std matching
multi-scale condition/stat matching
```

Then add final decoded image losses:

```text
image L1
LPIPS
DISTS
Stage3 anchor / fidelity guard
```

Promotion is based on decoded images, not condition_l1 alone.

## Priority 2: Condition Statistics and Diagnostics

Before changing architecture heavily, log condition statistics for:

```text
reference native condition
Stage3 x_aux native condition
predicted condition
condition delta
```

Metrics to log:

```text
mean/std/min/max
channel norm
per-channel variance
activation histograms
spatial spectrum
delta norm distribution
```

Purpose:

```text
detect scale mismatch
detect channel/stat mismatch
detect spatial-frequency mismatch
detect destructive residuals
detect weak base_condition
```

Full552 condition-stat diagnostic:

```text
run:
  20260628_stage4_condition_stats_full552

checkpoint:
  20260628_stage4_cod_lite_pyramid_sem256_detailctx6_tanh075_traincache2048_5k_b8.pt

split:
  Kodak24 + CLIC2020 test428 + DIV2K val100
  count: 552

actual_payload_bpp:
  0.013999

base_to_target_l1:
  0.5371

pred_to_target_l1:
  0.4112

pred L1 win rate over base:
  0.9982

base_to_target_relative_l2:
  0.8758

pred_to_target_relative_l2:
  0.6716

target / base / pred condition std:
  0.7954 / 0.8234 / 0.6932

target / base / pred spatial high-frequency ratio:
  0.2575 / 0.2776 / 0.2280
```

Interpretation:

```text
the adapter learns a real deterministic condition correction
mean alignment and cosine similarity improve substantially
the predicted condition is too low-energy and too low-frequency versus target
condition_l1 alone is not sufficient for decoded-image promotion
```

Immediate training implication:

```text
add condition cosine loss
add channel mean/std matching
add spatial spectrum or high-frequency-stat matching
monitor pred condition std/highfreq against target
keep delta residual bounded but avoid over-smoothing by tanh/scale
```

Condition-stat matching probe:

```text
run:
  20260628_stage4_cod_lite_pyramid_sem256_detailctx6_statsmatch_ft2k_b8

checkpoint:
  checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_statsmatch_ft2k_b8.pt

loss additions:
  condition cosine: 0.25
  condition channel stats: 0.20
  condition high-frequency ratio: 0.05
```

Full552 condition effect:

```text
pred_to_target_l1:
  0.4112 -> 0.4165

pred_to_target_cosine:
  0.7447 -> 0.7463

pred condition std:
  0.6932 -> 0.7282
  target: 0.7954

pred spatial high-frequency ratio:
  0.2280 -> 0.2349
  target: 0.2575
```

Full552 decoded-image effect at unchanged actual_payload_bpp:

```text
alpha0.30:
  PSNR     22.0432 -> 22.0421
  MS-SSIM   0.7360 ->  0.7377
  LPIPS     0.5421 ->  0.5372
  DISTS     0.3518 ->  0.3489
  patch-FID 112.76 -> 110.54

alpha1.00:
  PSNR     21.1677 -> 21.1181
  MS-SSIM   0.7064 ->  0.7125
  LPIPS     0.4487 ->  0.4428
  DISTS     0.3127 ->  0.3039
  patch-FID  65.16 ->  63.72
```

Interpretation:

```text
condition L1 can worsen while final decoded images improve, so decoded-image
promotion must remain the decision criterion. The probe supports condition
stat/spectrum matching as part of the curriculum, but it does not remove the
need for learned deterministic gating or larger training data.
```

## Priority 3: Deterministic Content Gate

A learned gate should replace fixed alpha diagnostics.

Inputs:

```text
x_aux
x_sem
residual magnitude
semantic latent statistics
detail code statistics/confidence
adapter delta norm
base condition quality proxy
```

Outputs:

```text
global alpha
spatial alpha map
condition-space channel/spatial gate
```

The gate must be deterministic from decoded payload and fixed model state. If a
gate/control map is transmitted, its bits must be counted in
`actual_payload_bpp`.

Promotion:

```text
beat fixed alpha0.25/0.30
no hidden reference leakage
no split-specific regression across Kodak24 / CLIC2020 test428 / DIV2K val100
```

The oracle adaptive-alpha headroom is useful but limited. Gate is a stabilizer;
large gains still require better condition recovery and better detail signals.

Current learned-gate probe:

```text
adapter:
  checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_statsmatch_ft2k_b8.pt

gate:
  checkpoints/stage4_cod_lite_gate/20260628_stage4_cod_lite_gate_statsmatch_fidelity_600_b1ga2.pt

eval:
  20260628_stage4_cod_lite_gate_statsmatch_fidelity_600_b1ga2_full552_eval

patch-FID audit:
  20260628_stage4_cod_lite_gate_statsmatch_fidelity_600_b1ga2_full552_patchfid256
```

Full552 result at unchanged actual_payload_bpp=0.013999:

```text
Stage 3:
  PSNR / MS-SSIM: 21.9951 / 0.7348
  LPIPS / DISTS: 0.5758 / 0.3536
  patch-FID256:   146.5134

learned deterministic gate:
  alpha mean/std/min/max: 0.3071 / 0.0773 / 0.1494 / 0.4414
  PSNR / MS-SSIM:        22.0367 / 0.7384
  LPIPS / DISTS:         0.5369 / 0.3481
  patch-FID256:          109.3629
```

Comparison to fixed alpha0.30 for the same stats-match adapter:

```text
fixed alpha0.30:
  PSNR / MS-SSIM: 22.0421 / 0.7377
  LPIPS / DISTS:  0.5372 / 0.3489
  patch-FID256:   110.5424

learned gate:
  slightly worse PSNR
  slightly better MS-SSIM, LPIPS, DISTS, and patch-FID
```

Interpretation:

```text
the gate path is valid and should replace further fixed-alpha hand tuning as
the main stabilization mechanism
the current gate is a Stage-4 internal mechanism, not a Stage-5 external
baseline win
next gate training should use larger train-cache data and stronger
content-dependent supervision/regularization
```

Implementation note:

```text
the official CoD-Lite image decode wrapper is batch-size-1 only in the current
image-loss path; use gradient accumulation for gate training until batch-safe
decode is implemented
```

## Priority 4: Diffusion-Friendly Detail Representation

Current detail residual mainly improves auxiliary reconstruction. Stage 4 needs
detail features that help diffusion condition recovery.

Initial no-extra-bit plan:

```text
same transmitted detail payload
  -> reconstruction detail feature for x_aux
  -> diffusion-control detail feature for adapter
```

Start safely:

```text
freeze entropy-coded detail stream
train diffusion-control detail projection head
then unfreeze detail decoder
then consider detail encoder only with actual entropy checks
```

Avoid immediately changing transmitted detail tokens and breaking bpp/roundtrip
stability.

## Priority 5: Stronger Adapter Architecture

Test stronger adapters before full backbone finetuning:

```text
multi-scale pyramid adapter
condition-space residual blocks
feature pyramid with skip connections
per-channel scale/gate
delta norm penalty
ControlNet-style side branch if hooks are available
FiLM / AdaGN modulation if hooks are available
```

Required CoSER-use ablations:

```text
no semantic latent
no detail context
no residual_hat
shuffled semantic latent
shuffled detail codes
zero adapter delta
base_condition only
```

Promotion requires decoded semantic/detail features to matter. If removing or
shuffling them does not hurt, CoSER is not using its own bitstream meaningfully.

## Priority 6: Scale Training Correctly

The current 2048-image train cache is too small for judging the ceiling.

Next target:

```text
20k to 100k 512-crop train cache if storage/time allow
OpenImages train
CLIC train
DIV2K train only with strict split hygiene
no Kodak / CLIC2020 test / DIV2K val leakage
```

Suggested curriculum:

```text
1. condition imitation pretraining
2. add image L1 + LPIPS + DISTS
3. add Stage3 anchor / edge / MS-SSIM guard
4. perceptual refinement
5. optional partial unfreeze or LoRA
```

Do not start with heavy perceptual loss without guard; previous runs already
show structure damage.

## Priority 7: Rate-Specific Backbones and CoD Track

Test CoD-Lite rate-specific checkpoints as decoder initialization:

```text
CoD_Lite_pretrain.pt
CoD_Lite_bpp_0_0078.pt
CoD_Lite_bpp_0_0156.pt
CoD_Lite_bpp_0_0312.pt
```

Use the same CoSER payload, same adapter protocol, and same evaluation split.
The CoD-Lite checkpoint is a fixed decoder/backbone initialization, not a CoSER
bitstream replacement.

Parallel heavy track:

```text
CoSER-DiC-Lite: CoD-Lite backbone
CoSER-DiC-CoD-1Step: one-step CoD backbone
CoSER-DiC-CoD-Heavy: multi-step CoD upper bound if feasible
```

Do not block CoD-Lite work on CoD.

## Priority 8: Stage 2/3 Rate Allocation and Entropy Models

The current semantic/detail split may not be Stage4-optimal.

Run actual-payload operating points:

```text
more semantic, less detail
less semantic, more detail
same semantic, diffusion-optimized detail
higher-quality semantic token, lower detail
lower semantic bpp, stronger detail residual
```

Evaluate Stage4, not only Stage3.

Improve entropy models:

```text
better semantic context prior
prefix-safe learned categorical prior
RDVQ-inspired top-k/escape ANS
semantic-conditioned residual Gaussian model
hyperprior or small side stream when counted
spatial/context residual prior
```

Freed bits can fund better detail or a tiny counted control stream.

## Priority 9: Tiny Counted Control Stream After Saturation

Only after deterministic adapter/gate saturates, consider very small
transmitted controls:

```text
condition residual tokens
content gate tokens
semantic confidence map
texture/refinement strength map
low-resolution alpha map
```

Rules:

```text
count bits in actual_payload_bpp
target 0.0005 to 0.002 bpp initially
compare curves / BD-rate, not one point
decode deterministically
```

## Evaluation Discipline

Do not claim BD-rate improvement from one point. Build curves.

Metrics:

```text
actual_payload_bpp / paper_bpp
estimated_bpp as diagnostic only
PSNR / MS-SSIM fidelity guard
LPIPS / DISTS
FID / KID with protocol-labeled patch settings
edge consistency
DINO similarity
segmentation consistency
OCR/text or small-object subset when available
visual failure audit
```

Baselines:

```text
official CoD-Lite .cod actual payload/file bpp
CoD / one-step CoD where reproducible
RDVQ real-bitstream bpp
GLC / StableCodec where reproducible
CompressAI anchors for RD context
```

Keep protocols clearly separated:

```text
CoD 512 reproduction:
  Kodak24 + CLIC2020 test428 + DIV2K val100

CoD-Lite README protocol:
  use official preprocessing and patch-FID settings

CoSER common LIC protocol:
  Kodak24 + CLIC Pro Valid41 + DIV2K val100
```

## Concrete Next Plan

Immediate:

```text
stop fixed-alpha-only optimization
keep alpha0.25/0.30 as stability anchors
log condition statistics
add semantic/detail ablations and shuffled-feature controls
scale train cache beyond 2048
evaluate full552 after each candidate
compare against official CoD-Lite curves
```

Next week:

```text
condition-imitation-first adapter
learned deterministic content gate
rate-specific CoD-Lite backbone initialization
stronger pyramid / condition-space residual adapter
at least 3 CoSER bpp operating points
official CoD-Lite full552 or matched-protocol baselines
```

After adapter saturation:

```text
partial unfreeze / LoRA
Stage3 diffusion-friendly detail head
tiny counted control stream
CoD / one-step CoD heavy-track ablation
joint Stage5 finetuning
```

## Win Condition Reframe

CoSER does not need to immediately beat CoD-Lite only by raw LPIPS at a single
bpp. A plausible win is:

```text
comparable perceptual quality
+ better fidelity / structure preservation
+ semantic/detail controllability
+ actual entropy-coded semantic/detail split
+ LPIPS/DISTS BD-rate or rate-perception-faithfulness gain
```

This must be measured, not asserted.

Expected trajectory:

```text
short term:
  Stage4 beats Stage3 more cleanly at same actual_payload_bpp

medium term:
  close part of the LPIPS/DISTS/FID gap to official CoD-Lite while preserving structure

long term:
  BD-rate or rate-perception-faithfulness improvement over CoD-Lite/CoD/RDVQ
```

The bottleneck is understandable: condition recovery and diffusion control are
weak. That does not invalidate the CoSER semantic/detail bitstream idea.
