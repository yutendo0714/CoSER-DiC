# Stage 4 CoD-Lite Adapter Bootstrap

Date: 2026-06-28 JST  
Status: Active Stage 4 implementation bootstrap  
Parent: `021_stage4_cod_codlite_parallel_backbone_policy.md`

## Decision

Start Stage 4 with a frozen official CoD-Lite one-step backbone and a
CoSER-owned condition adapter.

The adapter must not replace the CoSER semantic/detail bitstream with the
native CoD-Lite bitstream. It consumes decoder-available CoSER reconstructions:

```text
decoded x_aux / Stage 3 reconstruction
decoded x_sem / semantic-only reconstruction
decoded residual_hat = x_aux - x_sem
optional decoded semantic latent in later training
```

The decoder is allowed to run fixed CoD-Lite modules on `x_aux` because `x_aux`
is reconstructed from the transmitted CoSER bitstream and the CoD-Lite weights
are fixed decoder-side parameters. No image-specific CoD-Lite bitstream,
prompt, mask, seed, or table is transmitted.

## Implementation

Added:

```text
src/coserdic/models/gencodec_backbone.py
tests/test_gencodec_backbone.py
scripts/train_stage4_cod_lite_adapter.py
scripts/eval_stage4_cod_lite_adapter.py
scripts/eval_cod_lite_official_baseline.py
scripts/sweep_stage4_blend.py
```

Core wrapper:

```text
CoDLiteOneStepBackbone:
  loads official GenCodec/CoD_Lite one-step checkpoints
  freezes by default
  exposes native_condition(image)
  decodes from a supplied condition tensor
  maps official [-1, 1] output to [0, 1]

CoSERToCoDLiteConditionAdapter:
  predicts a CoD-Lite condition delta from decoded CoSER tensors
```

The first training script uses a native-stage3 base condition:

```text
base_cond:
  CoD-Lite native condition computed from decoded Stage 3 x_aux

target_cond:
  CoD-Lite native condition computed from the reference image
  training target only; not available at decode time

pred_cond:
  base_cond + adapter_delta(decoded CoSER tensors)
```

This is safer than predicting the entire condition from zero because a
zero-initialized adapter starts from a decoder-side transform of the decoded
Stage 3 reconstruction, not from an all-zero condition.

## Smoke Results

Wrapper load and random tensor smoke:

```text
checkpoint:
  external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.pt

condition:
  channels: 384
  128x128 input condition size: 8x8

result:
  real checkpoint loads through the CoSER wrapper
  native condition and adapter condition paths both decode without errors
```

Unit tests:

```text
.venv/bin/python -m pytest -q tests/test_gencodec_backbone.py tests/test_conditioning_adapter.py
result: 5 passed
```

Two-step native-base smoke:

```text
run:
  20260628_stage4_cod_lite_adapter_nativebase_smoke2_manifest128

manifest:
  strict 512 Stage 3 no-postprocess recon export

crop:
  128

result:
  checkpoint, samples, summary, and wandb offline run were written
```

Short Kodak256 probe:

```text
run:
  20260628_stage4_cod_lite_adapter_nativebase_probe20_kodak256

crop:
  256

max_steps:
  20

summary:
  condition_l1_mean: 0.550691
  image_l1_mean:     0.070614
  stage3_psnr_mean:  21.147420
  stage4_psnr_mean:  20.099120
```

Strict full512 probe:

```text
run:
  20260628_stage4_cod_lite_adapter_nativebase_probe100_full512

crop:
  512

max_steps:
  100

summary:
  condition_l1_mean: 0.502390
  image_l1_mean:     0.089350
  stage3_psnr_mean:  18.551640
  stage4_psnr_mean:  17.453740
```

LPIPS-loss smoke:

```text
run:
  20260628_stage4_cod_lite_adapter_nativebase_lpips020_probe20_kodak256

loss:
  condition_l1_weight: 0.5
  image_l1_weight:     0.25
  lpips_weight:        0.2

result:
  LPIPS fp32 backward path runs without NaNs
```

## Interpretation

Do not promote these probes as Stage 4 quality results.

What is validated:

```text
official CoD-Lite one-step checkpoint can be loaded in the CoSER codebase
frozen CoD-Lite decoder can run from CoSER-produced condition tensors
native-stage3 base condition avoids the unusable all-zero condition start
adapter checkpointing, samples, docs, and wandb logging work
512-crop training fits on the local GPU at batch size 1
LPIPS loss can be enabled without immediate NaNs
```

What is not validated:

```text
large improvement over Stage 3
large improvement over official CoD-Lite / CoD baselines
FID/KID improvement
semantic faithfulness under generative sharpening
full Stage 4 adapter design using decoded semantic latent tensors
```

The current samples show generative sharpening and hallucination, but structure
can be distorted. This confirms that Stage 4 must use a stronger promotion gate
than local training loss.

## Full512 Adapter Results

Two 5k-step full512 adapter runs were checked on Kodak24.

Perceptual-heavy run:

```text
train:
  20260628_stage4_cod_lite_adapter_nativebase_lpips020_full512_5k

eval:
  20260628_stage4_cod_lite_adapter_lpips020_full512_5k_kodak24_eval

actual_payload_bpp:
  0.014121

Stage 3:
  PSNR / MS-SSIM: 21.6672 / 0.7225
  LPIPS / DISTS: 0.6366 / 0.3732

Stage 4:
  PSNR / MS-SSIM: 20.2327 / 0.6711
  LPIPS / DISTS: 0.4756 / 0.2950

win rates:
  LPIPS 1.0
  DISTS 1.0
  PSNR 0.0
  MS-SSIM 0.0
```

This run strongly improves perceptual metrics but visibly damages structured
content.

L1-guarded run:

```text
train:
  20260628_stage4_cod_lite_adapter_nativebase_lpips005_l1guard_full512_5k

eval:
  20260628_stage4_cod_lite_adapter_lpips005_l1guard_full512_5k_kodak24_eval_save24

actual_payload_bpp:
  0.014121

Stage 3:
  PSNR / MS-SSIM: 21.6672 / 0.7225
  LPIPS / DISTS: 0.6366 / 0.3732
  GenCodec Kodak patch-FID: 216.7101

Stage 4:
  PSNR / MS-SSIM: 20.5132 / 0.6822
  LPIPS / DISTS: 0.4964 / 0.3151
  GenCodec Kodak patch-FID: 117.5225
```

This run is healthier than the perceptual-heavy run but still loses structure
and fidelity. It is not promoted as a Stage 4 endpoint.

Deterministic decoder-side blend sweep:

```text
payload policy:
  stage4 = (1 - alpha) * stage3 + alpha * adapter_output
  fixed alpha is decoder-side configuration
  no image-specific side information is transmitted

runs:
  20260628_stage4_l1guard_decoder_blend_sweep_kodak24
  20260628_stage4_lpips020_decoder_blend_sweep_kodak24
```

Best safe Kodak24 point:

```text
source adapter:
  20260628_stage4_cod_lite_adapter_nativebase_lpips020_full512_5k

alpha:
  0.1

actual_payload_bpp:
  0.014121

Stage 3:
  PSNR / MS-SSIM: 21.6672 / 0.7225
  LPIPS / DISTS: 0.6366 / 0.3732
  GenCodec Kodak patch-FID: 216.7101

Stage 4 alpha0.1:
  PSNR / MS-SSIM: 21.6785 / 0.7242
  LPIPS / DISTS: 0.6204 / 0.3725
  GenCodec Kodak patch-FID: 193.2827
```

Strict 512 full split confirmation:

```text
run:
  20260628_stage4_cod_lite_adapter_lpips020_alpha010_full552_eval

dataset:
  Kodak24 + CLIC2020 test428 + DIV2K val100
  total: 552 images

actual_payload_bpp:
  0.013999

Stage 3:
  PSNR / MS-SSIM: 21.9951 / 0.7348
  LPIPS / DISTS: 0.5758 / 0.3536

Stage 4 alpha0.1:
  PSNR / MS-SSIM: 22.0147 / 0.7365
  LPIPS / DISTS: 0.5626 / 0.3529

win rates:
  PSNR:    0.7627
  MS-SSIM: 0.9601
  LPIPS:   0.9982
  DISTS:   0.5507
```

This is promoted only as the current safe internal Stage 4 anchor. It improves
Stage 3 means without extra payload, but it does not yet beat official
CoD-Lite pretrained checkpoints on perceptual quality.

## Official CoD-Lite Baseline Anchor

Official CoD-Lite rate checkpoints were evaluated with real `.cod`
compress/decompress payload bpp on the same Kodak24 512 inputs.

```text
CoD-Lite 0.0039: actual_payload_bpp 0.003906, LPIPS 0.3390, DISTS 0.2089, FID 51.2190
CoD-Lite 0.0078: actual_payload_bpp 0.007812, LPIPS 0.2807, DISTS 0.1733, FID 44.6564
CoD-Lite 0.0156: actual_payload_bpp 0.015625, LPIPS 0.2259, DISTS 0.1402, FID 38.4785
CoD-Lite 0.0312: actual_payload_bpp 0.031250, LPIPS 0.1614, DISTS 0.1120, FID 31.8072
```

Full baseline memo:

```text
docs/research/baselines/cod_lite_official_baseline_20260628.md
```

Conclusion:

```text
CoSER Stage 4 currently improves Stage 3 perceptual/distribution metrics but
does not yet beat official CoD-Lite pretrained checkpoints at comparable bpp.
The next Stage 4 iteration must be judged against this curve, not only against
Stage 3. The alpha0.1 deterministic blend is the current stability baseline.
```

## Next Experiments

Recommended next design iteration:

```text
1. Keep the official CoD-Lite / CoD decoder-side prior, but reduce destructive
   condition deltas.
2. Add an explicit Stage 3 identity/structure guard:
   - learned deterministic decoder-side gate from decoded tensors
   - delta-norm penalty on condition updates
   - edge/text/structure-weighted reconstruction guard
3. Evaluate every candidate against:
   - Stage 3 no-postprocess
   - Stage 3 fixed postprocess operating points
   - official CoD-Lite rate checkpoint curve
   - CoD one-step or pixel/latent baseline when reproducible
4. Promote only if actual_payload_bpp is unchanged or explicitly counted and
   perceptual gains do not come from unacceptable semantic/structural drift.
```
