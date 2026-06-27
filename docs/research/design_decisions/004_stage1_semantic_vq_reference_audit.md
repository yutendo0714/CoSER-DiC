# Stage 1 Semantic VQ Official-Implementation Audit

Date: 2026-06-27  
Status: Active Stage 1 freeze gate  
Parent policy: `docs/research/design_decisions/003_official_implementation_reference_policy.md`

## Trigger

Stage 1 already has a promising semantic VQ checkpoint, but that is not enough
to freeze it. The Core-MVP policy requires official implementations to be used
from Stage 1 onward, not only after Stage 2 starts.

This audit checks whether the current Stage 1 semantic tokenizer is still the
right CoSER-owned implementation after reading the mapped official references.

## References Checked

```text
external/repos/RDVQ/tokenizer/tokenizer_image/models/quantizer.py
external/repos/RDVQ/tokenizer/tokenizer_image/training/losses/vq_loss.py
external/repos/GLC/src/models/image_model.py
external/repos/GenCodec/CoD_Lite/cod/models/condition_codec.py
```

## Current CoSER Stage 1 Files

```text
src/coserdic/models/semantic_vq.py
src/coserdic/losses/stage1.py
scripts/train_stage1_semantic_vq.py
scripts/analyze_stage1_semantic_vq.py
scripts/eval_stage1_semantic_bitstream.py
src/coserdic/entropy/semantic_tokens.py
src/coserdic/entropy/bitstream.py
```

## Findings

### 1. Keep Stage 1 CoSER-owned and scratch-trained

RDVQ and CoD-Lite both confirm that a compact VQ token stream is a reasonable
semantic or conditioning interface, but directly replacing CoSER Stage 1 with
RDVQ/CoD-Lite would erase the Core-MVP claim. The current CoSER tokenizer stays
as a scratch-trained semantic VQ stream.

Adopt:

```text
VQ-VAE-style distance assignment
straight-through quantization
explicit codebook/commitment decomposition
code usage and token entropy diagnostics
actual transmitted-byte token audit
```

Do not adopt by default:

```text
RDVQ checkpoint initialization
RDVQ tokenizer architecture as the CoSER tokenizer
CoD-Lite condition codec as the primary semantic tokenizer
GLC VQGAN latent codec as a replacement for semantic/detail separation
```

### 2. Treat current usage loss as anti-collapse, not RD entropy optimization

RDVQ uses entropy-aware VQ machinery and conditional entropy hooks. The current
CoSER Stage 1 `usage_loss` instead encourages broad codebook use to prevent
early collapse.

This is acceptable for Stage 1 stabilization, but it must not be presented as
the final entropy model or final rate term.

Action taken:

```text
Log RDVQ-style entropy diagnostics:
  assignment_sample_entropy_bits
  assignment_avg_entropy_bits
  soft_usage_entropy_bits

Log VQ loss decomposition:
  loss_vq_commitment
  loss_vq_codebook
```

### 3. Keep perceptual/GAN losses out of the default Stage 1 route for now

RDVQ's training loss includes perceptual and GAN terms. CoSER probes showed
LPIPS/MS-SSIM-style losses can introduce non-finite gradients or destabilize
hard-VQ warmup at Stage 1.

Decision:

```text
Default Stage 1 remains L1 + low-pressure VQ + weak usage pressure.
Perceptual/GAN losses are ablations after the stable tokenizer route is fixed.
```

### 4. Fixed-bit packing remains a Stage 1 audit baseline

CoD-Lite's condition codec confirms that simple exact-width bit packing is a
useful debugging and payload audit tool. CoSER's fixed-bit semantic token codec
is valid for Stage 1 auditing, but it is not the final entropy-coded semantic
stream.

Action taken:

```text
Stage 1 bitstream evaluation records semantic token count from the actual
encoder output rather than assuming crop_size // 32.
```

### 5. GLC lessons are mostly Stage 3 lessons

GLC is important because it couples VQ/generative latent coding with
hyperprior/spatial-prior residual coding. That lesson is relevant to CoSER's
detail residual stream, but Stage 1 should not become a GLC-style full latent
codec.

## Verification

```text
pytest -q
  16 passed

20260627_stage1_reference_audit_metrics_smoke
  max_steps: 2
  status: passed
  wandb: wandb/offline-run-20260627_091543-gh2qvnt3
```

## Stage 1 Freeze Gate

Before treating the current Stage 1 checkpoint as fixed for Stage 2, run and
document:

```text
1. Kodak hard-VQ analysis with entropy diagnostics.
2. Stage 1 actual bitstream eval with fixed_bits and zlib_fixed_bits.
3. A short smoke run confirming new diagnostics log to W&B.
4. A negative/positive decision on longer 10k-20k low-pressure Stage 1 training.
```

The Stage 1 checkpoint can be used for Stage 2 only if:

```text
token roundtrip is exact
actual payload and full-stream bpp are reported
semantic-only quality is documented
code usage does not collapse
the remaining gap to the continuous AE is acknowledged
```
