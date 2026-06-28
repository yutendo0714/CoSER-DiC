# Stage 4 CoD / CoD-Lite Parallel Backbone Policy

Date: 2026-06-28 JST  
Status: Active Stage 4 policy  
Parent:
`docs/research/design_decisions/000_external_repos_and_pretrained_policy_mvp.md`,
`docs/research/design_decisions/003_official_implementation_reference_policy.md`

## Decision

Reset the Stage 4 decoder-side ResUNet refiner results as non-binding
diagnostics for the Core MVP path.

The Core MVP Stage 4 path returns to compression-oriented diffusion decoding:

```text
default lightweight track:
  CoD-Lite pretrained backbone
  external/pretrained/CoD_Lite/CoD_Lite_pretrain.pt

parallel heavy track:
  CoD pretrained / one-step checkpoints
  external/pretrained/CoD/*
```

CoD-Lite remains the default implementation target because it is lightweight,
one-step, and better aligned with the MVP's practical decoder. CoD is allowed
in parallel as a heavier upper-bound / teacher / backbone candidate because CoD
and CoD-Lite are part of the same GenCodec family and expose different points
on the quality-cost curve.

## Why Not Keep Optimizing The ResUNet Refiner

The ResUNet refiner was useful only as a scaffold:

```text
validated:
  decoder-side fixed weights do not add image-specific payload bpp
  Stage 3 actual-bitstream evaluation can produce Stage 4 reconstructions
  x_aux / x_sem / residual / semantic latent conditioning can be wired through
  full CLIC2020 + DIV2K patch-FID gate is necessary

not validated:
  CoD/CoD-Lite diffusion behavior
  diffusion sampling or one-step generative reconstruction
  adapter design for GenCodec backbones
  ability to beat CoD/CoD-Lite baselines
```

Therefore, ResUNet results should not steer Stage 4 architecture choices except
as a warning that local LPIPS/DISTS gains can regress distribution metrics.

## Stage 4 Definition

Stage 4 is not a generic learned postprocessor. It is:

```text
Stage 3 actual bitstream
  -> x_aux / Stage 3 reconstruction
  -> CoSER conditioning adapter
  -> CoD-Lite or CoD diffusion backbone
  -> x_hat
```

The adapter must be CoSER-specific and trained from scratch:

```text
inputs:
  x_aux or Stage 3 reconstruction
  semantic token features
  decoded semantic latent
  decoded detail residual / detail latent
  optional fixed rate/perception condition

outputs:
  conditioning tensors / FiLM / control features for the diffusion backbone
```

The diffusion backbone may be frozen initially. Any per-image prompt, mask,
map, stochastic seed, or control payload required by the decoder must be counted
in `actual_payload_bpp`. Fixed decoder weights and fixed pretrained priors are
not counted.

## Stage 5 Definition

Stage 5 is joint Core MVP finetuning after a Stage 4 diffusion path works:

```text
train jointly:
  semantic encoder
  detail residual encoder
  entropy models / priors
  auxiliary decoder
  diffusion adapter
  optionally very-low-lr diffusion backbone

keep:
  actual compress/decompress evaluation
  semantic-only / no-detail / no-diffusion ablations
  actual_payload_bpp as the paper bpp
```

Do not claim Stage 5 until baseline runners and ablations are available or
explicitly waived.

## Parallel Track Roles

```text
CoD-Lite track:
  primary MVP implementation path
  first adapter integration target
  expected to be trainable/evaluable on the local machine
  best candidate for fast iteration and eventual practical codec

CoD track:
  upper-bound / teacher / heavy backbone path
  may provide better perceptual ceiling
  costs several GB per checkpoint and likely more GPU memory/runtime
  should be evaluated in parallel when assets and memory permit
```

The project should compare:

```text
Stage 3 x_aux only
CoD-Lite adapter x_hat
CoD adapter x_hat
official CoD-Lite rate checkpoint baseline
official CoD rate checkpoint / DiffC baseline if reproducible
```

## Immediate Next Steps

```text
done:
  Verify/download CoD-Lite pretrain and selected CoD configs/checkpoints.
  Local verification is recorded in:
    docs/research/baselines/pretrained_asset_inventory_20260628.md

next:
1. Inspect official GenCodec model loaders and checkpoint key structure.
2. Implement a CoSER diffusion backbone wrapper under src/coserdic.
3. Start with adapter-only smoke training, backbone frozen.
4. Evaluate actual Stage 3 payload bpp + diffusion x_hat metrics.
5. Promote only if full Kodak24 + CLIC2020 test428 + DIV2K val100 improves
   perceptual metrics without unacceptable FID/semantic drift.
```

## Non-Goals

```text
Do not:
  replace CoSER semantic/detail streams with native CoD/CoD-Lite bitstreams
  call official CoD-Lite rate checkpoints "CoSER-DiC"
  use forward-estimated bpp as the paper metric
  let the ResUNet refiner decide the final Stage 4 architecture
```
