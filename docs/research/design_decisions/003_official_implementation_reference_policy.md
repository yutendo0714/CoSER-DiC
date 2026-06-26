# Official Implementation Reference Policy for CoSER-DiC Core MVP

Date: 2026-06-27
Status: Active implementation policy

## Conclusion

CoSER-DiC should remain a new codec implemented under `src/coserdic`, not a
fork or direct assembly of existing official repositories.

At the same time, the Core MVP combines design families that already have
important implementation lessons:

- RD-aware / entropy-aware VQ tokenization
- actual entropy-coded bitstreams
- low-bitrate generative latent coding
- one-step compression-oriented diffusion
- hyperprior / autoregressive entropy coding

Therefore, official implementations must be used as implementation references,
baselines, sanity checks, and limited initialization sources. They must not be
treated as the CoSER-DiC method itself.

## Reference Levels

Use these levels when adding or reviewing code:

```text
L0 baseline-only:
  The external repo is only executed/evaluated as a comparison method.

L1 conceptual reference:
  Architecture or training idea informs CoSER design, but code is independent.

L2 implementation reference:
  Tensor shapes, coding flow, loss decomposition, and evaluation protocol are
  studied and reimplemented in CoSER style.

L3 wrapped tool or initialization:
  A pretrained backbone or codec utility is used behind a CoSER-specific
  adapter. Provenance and config must be logged.

L4 copied/adapted code:
  Avoid by default. Allowed only when license-compatible, clearly attributed,
  localized, tested, and documented.
```

Default Core MVP rule:

```text
semantic VQ: scratch CoSER implementation, official repos as L1-L2 reference
detail residual: scratch CoSER implementation, official repos as L1-L2 reference
entropy models: scratch CoSER implementation, official repos as L1-L2 reference
diffusion decoder: CoD-Lite pretrained may be L3 initialization/backbone
baselines: official repos can be L0 wrappers/evaluators
```

## Local Reference Snapshot

These local clones were inspected for this policy.

| Repo | Path | Commit | License | Core Role |
|---|---|---:|---|---|
| RDVQ | `external/repos/RDVQ` | `a2744eedd803d9e20b55b4b0c935ec211a07d184` | MIT | RD-aware VQ and real entropy coding reference |
| GenCodec / CoD-Lite | `external/repos/GenCodec` | `c49eb0d643cc75e6c732cbc311a508627b54cf06` | MIT | one-step diffusion codec reference and initialization candidate |
| GLC | `external/repos/GLC` | `126db25f9c093508cd0c99cee32b53fd60074f9a` | Apache-2.0 | VQ generative latent codec reference |
| StableCodec | `external/repos/StableCodec` | `73d9597cd97b8bca99c6c01a0156c37e6040e643` | MIT | one-step diffusion codec baseline/fallback reference |
| CompressAI | `.venv` package | installed package | BSD-style package | entropy model and actual compress/decompress anchor |

CompressAI paths:

```text
.venv/lib/python3.10/site-packages/compressai/__init__.py
.venv/lib/python3.10/site-packages/compressai/entropy_models/entropy_models.py
```

## Current CoSER Implementation Status

Current Stage 1 semantic VQ is mostly a CoSER-specific implementation:

```text
src/coserdic/models/semantic_vq.py
src/coserdic/losses/stage1.py
scripts/train_stage1_semantic_vq.py
scripts/analyze_stage1_semantic_vq.py
scripts/eval_stage1_semantic_bitstream.py
src/coserdic/entropy/semantic_tokens.py
src/coserdic/entropy/bitstream.py
```

What it currently borrows conceptually:

```text
VQ-VAE-style encoder / codebook / straight-through quantization
actual transmitted-byte evaluation discipline
low-bitrate token payload auditing
```

What it does not yet borrow directly:

```text
RDVQ conditional entropy predictor
RDVQ top-k/escape tensor-rANS coder
GLC hyperprior/spatial-prior residual coding
CoD-Lite one-step diffusion backbone
StableCodec latent codec/diffusion path
CompressAI EntropyBottleneck/GaussianConditional modules
```

This is acceptable for Stage 1 stabilization, but it is not sufficient for the
full Core MVP. Stage 2 and later must be guided by official implementations.

## Component Reference Map

### Stage 1 Semantic VQ Token Stream

CoSER files:

```text
src/coserdic/models/semantic_vq.py
src/coserdic/losses/stage1.py
scripts/train_stage1_semantic_vq.py
```

Primary references:

```text
external/repos/RDVQ/tokenizer/tokenizer_image/models/quantizer.py
external/repos/RDVQ/tokenizer/tokenizer_image/training/losses/vq_loss.py
external/repos/GLC/src/models/image_model.py
external/repos/GenCodec/CoD_Lite/cod/models/condition_codec.py
```

Reference lessons:

```text
RDVQ:
  - separates vq_loss and commit_loss
  - supports entropy-aware VQ pressure
  - tracks codebook usage
  - has conditional entropy predictor hooks

GLC:
  - codes a VQGAN latent rather than pixels
  - couples hyperprior/spatial-prior residual coding with generative latent

CoD-Lite condition codec:
  - has simple fixed-bit token packing
  - uses lightweight codec blocks for one-step diffusion conditioning
```

CoSER decision:

```text
Keep the Stage 1 semantic tokenizer scratch-trained.
Do not initialize from RDVQ by default.
Do not turn Stage 1 into an RDVQ tokenizer.
Use RDVQ's entropy-aware VQ and real-coding discipline as Stage 2 references.
```

### Stage 2 Semantic Token Entropy Model

CoSER target files:

```text
src/coserdic/entropy/
scripts/eval_stage1_semantic_bitstream.py
future: src/coserdic/models/semantic_token_prior.py
future: scripts/train_stage2_semantic_entropy.py
future: scripts/eval_stage2_semantic_entropy_bitstream.py
```

Primary references:

```text
external/repos/RDVQ/tokenizer/tokenizer_image/codec/real/simple.py
external/repos/RDVQ/tokenizer/tokenizer_image/codec/real/causal_pipeline.py
external/repos/RDVQ/tokenizer/tokenizer_image/codec/entropy_coding/codecs/topk_tensor_rans.py
external/repos/RDVQ/tokenizer/tokenizer_image/codec/entropy_coding/probability.py
external/repos/RDVQ/tokenizer/tokenizer_image/codec/entropy_coding/symbol_mapping.py
.venv/lib/python3.10/site-packages/compressai/entropy_models/entropy_models.py
```

Reference lessons:

```text
RDVQ:
  - decoder must rebuild logits/CDFs from decoded causal history
  - top-k/escape coding avoids full codebook CDF cost
  - actual payload bits must be measured from encoded byte streams

CompressAI:
  - model.update(force=True) before compress/decompress
  - actual bpp should count returned byte strings, not likelihood estimates
```

CoSER decision:

```text
Implement CoSER semantic token prior independently.
Use RDVQ top-k/escape tensor-rANS as the main L2 reference.
Keep fixed_bits as a deterministic audit baseline, not the final entropy model.
Official metrics must report actual encoded bytes.
```

### Detail Residual Latent Stream

CoSER target files:

```text
future: src/coserdic/models/detail_residual.py
future: src/coserdic/models/residual_entropy_model.py
future: scripts/train_stage3_detail_residual.py
```

Primary references:

```text
external/repos/GLC/src/models/image_model.py
external/repos/GLC/src/models/common_model.py
external/repos/StableCodec/ELIC/model/elic_official.py
.venv/lib/python3.10/site-packages/compressai/entropy_models/entropy_models.py
```

Reference lessons:

```text
GLC:
  - residual transform in generative latent space
  - hyperprior plus spatial prior over latent residual
  - quantization-step parameters for rate control

StableCodec / ELIC:
  - learned residual entropy coding for diffusion-conditioned reconstruction

CompressAI:
  - robust implementation patterns for hyperprior and Gaussian conditionals
```

CoSER decision:

```text
Detail residual must be conditioned on semantic tokens/aux reconstruction.
It should not duplicate information already represented by semantic VQ tokens.
```

### One-Step Diffusion Reconstruction

CoSER files:

```text
src/coserdic/models/diffusion_decoder.py
src/coserdic/models/conditioning_adapter.py
configs/train/train_stage4_diffusion.yaml
```

Primary references:

```text
external/repos/GenCodec/CoD_Lite/finetuned_one_step_codec/models/models.py
external/repos/GenCodec/CoD_Lite/cod/diffusion/diffusion.py
external/repos/GenCodec/CoD_Lite/cod/models/pixel_dit.py
external/repos/GenCodec/CoD_Lite/cod/models/condition_codec.py
external/repos/StableCodec/src/StableCodec.py
external/repos/StableCodec/src/compress.py
```

Reference lessons:

```text
CoD-Lite:
  - one-step inference from noise plus codec conditioning
  - pretrained denoiser weights can initialize the diffusion backbone
  - condition codec is not natively CoSER semantic/detail conditioning

StableCodec:
  - actual compress/decompress API around a diffusion codec
  - image padding/cropping and bpp accounting patterns
```

CoSER decision:

```text
Use CoD_Lite_pretrain.pt only as a backbone initialization candidate.
Implement CoSER-specific conditioning adapter for x_aux, semantic tokens,
detail latents, and rate/perception controls.
Do not treat CoD-Lite rate checkpoints as CoSER-DiC models.
```

### Actual Bitstream and Evaluation

CoSER files:

```text
src/coserdic/entropy/bitstream.py
src/coserdic/entropy/semantic_tokens.py
scripts/eval_stage1_semantic_bitstream.py
scripts/eval_compressai_anchor.py
```

Primary references:

```text
external/repos/RDVQ/utils/bitstream_container.py
external/repos/RDVQ/utils/real_codec_stats.py
external/repos/RDVQ/real_codec_inference.py
external/repos/StableCodec/src/compress.py
scripts/eval_compressai_anchor.py
```

Reference lessons:

```text
Every official result should include:
  - bytes actually emitted by compress
  - successful decompress / token roundtrip
  - bpp from transmitted bytes
  - clear distinction between payload bpp and container bpp
```

CoSER decision:

```text
Keep CoSERBitstream as a transparent container for auditability.
Optimize payload entropy coding separately from outer-container overhead.
Report both when useful, but official codec bpp must be actual transmitted bpp.
```

## Immediate Implementation Checklist

Before implementing each next stage:

```text
1. Read the mapped official files for that stage.
2. Add a short design note saying which implementation lessons are adopted.
3. Keep CoSER-specific data flow explicit.
4. Add actual compress/decompress or roundtrip tests.
5. Log reference repo commit hashes in experiment docs.
```

Next recommended work:

```text
1. Build a minimal semantic token prior for Stage 2.
2. Use RDVQ top-k/escape tensor-rANS as the implementation reference.
3. Compare actual payload bpp against Stage 1 fixed_bits = 0.01270 bpp.
4. Keep the current best Stage 1 checkpoint fixed while testing entropy coding.
5. In parallel, audit CoD-Lite checkpoint key compatibility for Stage 4.
```

