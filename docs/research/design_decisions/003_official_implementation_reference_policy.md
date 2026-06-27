# Official Implementation Reference Policy for CoSER-DiC Core-MVP

Date: 2026-06-27  
Status: Active implementation policy  
Recommended path: `docs/research/design_decisions/001_core_mvp_implementation_reference_policy.md`

## 0. Scope and meaning of Core-MVP

In this document, **Core-MVP** means the **minimal complete implementation of the CoSER-DiC core method**.

It is not a toy prototype, not a temporary placeholder, and not a simplified method that will later be replaced by the “real” CoSER-DiC.

The Core-MVP must contain the components required to test the central CoSER-DiC claim:

```text
Extreme-low-bitrate generative image compression benefits from separating
semantic structure, detail-critical residuals, and stochastic texture generation
under entropy-constrained coding.
```

Therefore, the Core-MVP must eventually include:

```text
1. Semantic VQ token stream
2. Entropy-aware semantic token coding
3. Detail residual latent stream
4. Semantic-conditioned residual entropy model
5. Auxiliary reconstruction
6. Compression-oriented one-step or few-step diffusion reconstruction
7. CoSER-specific conditioning adapter
8. Actual entropy-coded bitstream
9. compress/decompress API
10. actual bpp reporting
11. semantic-only / no-residual / no-diffusion ablations
```

Stage 1 alone is not the Core-MVP.  
Stage 1 is the semantic VQ stabilization stage.  
The Core-MVP is complete only after Stage 5.

---

## 1. Conclusion

CoSER-DiC should remain a new codec implemented under `src/coserdic`, not a fork or direct assembly of existing official repositories.

At the same time, the Core-MVP combines design families that already have important implementation lessons:

```text
- RD-aware / entropy-aware VQ tokenization
- actual entropy-coded bitstreams
- low-bitrate generative latent coding
- semantic/detail residual decomposition
- one-step compression-oriented diffusion
- hyperprior / autoregressive entropy coding
```

Therefore, official implementations must be used as implementation references, baselines, sanity checks, and limited initialization sources.

They must not be treated as the CoSER-DiC method itself.

---

## 2. Reference levels

Use these levels when adding or reviewing code:

```text
L0 baseline-only:
  The external repo is only executed/evaluated as a comparison method.

L1 conceptual reference:
  Architecture or training idea informs CoSER design, but code is independent.

L2 implementation reference:
  Tensor shapes, coding flow, loss decomposition, entropy-coding flow, and
  evaluation protocol are studied and reimplemented in CoSER style.

L3 wrapped tool or initialization:
  A pretrained backbone or codec utility is used behind a CoSER-specific
  adapter. Provenance and config must be logged.

L4 copied/adapted code:
  Avoid by default. Allowed only when license-compatible, clearly attributed,
  localized, tested, and documented.
```

Default Core-MVP rule:

```text
semantic VQ:
  scratch CoSER implementation;
  official repos as L1-L2 references.

detail residual:
  scratch CoSER implementation;
  official repos as L1-L2 references.

entropy models:
  scratch CoSER implementation;
  official repos as L1-L2 references.

diffusion decoder:
  CoD-Lite pretrained may be L3 initialization/backbone.

baselines:
  official repos can be L0 wrappers/evaluators.
```

The only default pretrained model used inside the Core-MVP should be:

```text
CoD_Lite_pretrain.pt
```

and only for the diffusion reconstruction backbone, if feasible.

RDVQ, GLC, Control-GIC, StableCodec, AEIC, OneDC, ResULIC, and RDEIC are references and baselines unless a later ablation explicitly promotes a component.

---

## 3. Local reference snapshot

These local clones were inspected for this policy.

| Repo | Path | Commit | License | Core role |
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

If any repository commit changes, update this table before reporting results.

---

## 4. Current CoSER implementation status

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

This is acceptable for Stage 1 stabilization, but it is not sufficient for the full Core-MVP. Stage 2 and later must be guided by official implementations more strongly.

---

# 5. Component reference map

## 5.1 Stage 1 Semantic VQ Token Stream

CoSER files:

```text
src/coserdic/models/semantic_vq.py
src/coserdic/losses/stage1.py
scripts/train_stage1_semantic_vq.py
scripts/analyze_stage1_semantic_vq.py
```

Primary references:

```text
external/repos/RDVQ/tokenizer/tokenizer_image/models/quantizer.py
external/repos/RDVQ/tokenizer/tokenizer_image/training/losses/vq_loss.py
external/repos/GLC/src/models/image_model.py
```

Secondary reference:

```text
external/repos/GenCodec/CoD_Lite/cod/models/condition_codec.py
```

Reference lessons:

```text
RDVQ:
  - separates vq_loss and commit_loss
  - supports entropy-aware VQ pressure
  - tracks codebook usage
  - has conditional entropy predictor hooks
  - provides real-coding discipline for later stages

GLC:
  - codes a VQGAN/generative latent rather than pixels
  - provides low-bitrate VQ/generative-latent design cues
  - couples hyperprior/spatial-prior residual coding with generative latent

CoD-Lite condition codec:
  - has simple fixed-bit token packing
  - uses lightweight codec blocks for one-step diffusion conditioning
  - should be treated as a secondary reference at Stage 1
```

Important clarification:

```text
CoD-Lite condition codec is a secondary reference for payload auditing and
lightweight conditioning structure. It is not the primary reference for
semantic VQ learning.
```

CoSER decision:

```text
Keep the Stage 1 semantic tokenizer scratch-trained.
Do not initialize from RDVQ by default.
Do not turn Stage 1 into an RDVQ tokenizer.
Use RDVQ's entropy-aware VQ and real-coding discipline as Stage 2 references.
```

Stage 1 expected output:

```text
semantic tokens
semantic embeddings
semantic-only reconstruction x_sem
perplexity
dead_code_ratio
assignment histogram
fixed-bits semantic token payload audit
```

---

## 5.2 Stage 2 Semantic Token Entropy Model

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
  - entropy model APIs should be explicit about update, compress, decompress
```

CoSER decision:

```text
Implement CoSER semantic token prior independently.
Keep fixed_bits as a deterministic audit baseline, not the final entropy model.
Official metrics must report actual encoded bytes.
```

Recommended implementation order:

```text
Step 1:
  Implement a minimal CoSER-owned semantic token prior.

Recommended starting options:
  - factorized categorical prior
  - simple context-free or grouped categorical prior
  - hyperprior-conditioned categorical prior, if simple prior is insufficient

Step 2:
  Add actual byte-stream evaluation around this prior.

Step 3:
  Use RDVQ top-k/escape tensor-rANS as the target L2 reference for a more
  practical actual entropy-coded version.

Step 4:
  Only after simple actual coding works, consider stronger causal/context
  priors.
```

Important clarification:

```text
RDVQ top-k/escape tensor-rANS is the main target reference for the actual
entropy-coded Stage 2 version, but it does not need to be the first
implementation step.
```

Avoid initially:

```text
full masked Transformer prior
Mamba/SSM token prior
large autoregressive prior that blocks debugging
```

Stage 2 expected output:

```text
semantic_token_bpp_estimated
semantic_token_bpp_actual
semantic_token_stream
semantic_hyper_stream, if used
semantic token roundtrip test
estimated-vs-actual bitrate report
```

---

## 5.3 Stage 3 Detail Residual Latent Stream

CoSER target files:

```text
future: src/coserdic/models/detail_residual.py
future: src/coserdic/models/residual_entropy_model.py
future: src/coserdic/models/joint_aux_decoder.py
future: scripts/train_stage3_detail_residual.py
```

Primary implementation references:

```text
external/repos/GLC/src/models/image_model.py
external/repos/GLC/src/models/common_model.py
external/repos/StableCodec/ELIC/model/elic_official.py
.venv/lib/python3.10/site-packages/compressai/entropy_models/entropy_models.py
```

Additional conceptual references:

```text
ResULIC:
  - semantic residual coding motivation
  - compression-aware diffusion conditioning

RDEIC:
  - relay residual diffusion
  - residual refinement and diffusion-conditioned reconstruction

DLF:
  - semantic/detail dual-branch decomposition
  - separation between semantic structure and detail-critical information
```

Reference lessons:

```text
GLC:
  - residual transform in generative latent space
  - hyperprior plus spatial prior over latent residual
  - quantization-step parameters for rate control

StableCodec / ELIC:
  - learned residual entropy coding for diffusion-conditioned reconstruction
  - auxiliary encoder/decoder and diffusion handoff patterns

CompressAI:
  - robust implementation patterns for hyperprior and Gaussian conditionals
  - actual compress/decompress conventions for continuous latents
```

CoSER decision:

```text
Detail residual must be conditioned on semantic tokens and/or auxiliary reconstruction.
It should not duplicate information already represented by semantic VQ tokens.
It should not become a full transform codec that dominates total bitrate.
```

Recommended starting point:

```text
detail encoder input:
  concat[x, x_sem, x - x_sem, abs(x - x_sem)]

detail prior:
  semantic-conditioned Gaussian conditional with hyperprior

initial distribution:
  single Gaussian, not Gaussian mixture

initial quantization:
  scalar quantization or additive uniform noise during training
```

Stage 3 expected output:

```text
detail_latent_bpp_estimated
detail_latent_bpp_actual, if actual coding is ready
joint auxiliary reconstruction x_aux
semantic-only vs semantic+detail comparison
detail residual energy map
semantic/detail bpp decomposition
```

---

## 5.4 Stage 4 One-Step Diffusion Reconstruction

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

CoD:
  - compression-oriented diffusion pretraining concept
  - possible teacher/foundation model role, not Core-MVP default backbone

StableCodec:
  - actual compress/decompress API around a diffusion codec
  - image padding/cropping and bpp accounting patterns
  - fallback one-step diffusion codec reference

OneDC / AEIC:
  - additional one-step / asymmetric generative compression references
  - baseline references, not Core-MVP defaults
```

CoSER decision:

```text
Use CoD_Lite_pretrain.pt only as a backbone initialization candidate.
Implement CoSER-specific conditioning adapter for:
  - x_aux
  - semantic token features
  - detail latent features
  - rate/perception controls

Do not treat CoD-Lite rate checkpoints as CoSER-DiC models.
Do not make StableCodec the default decoder unless CoD-Lite integration fails
and that decision is documented.
```

Recommended starting point:

```text
1. Load CoD_Lite_pretrain.pt and verify checkpoint key compatibility.
2. Freeze most of the diffusion backbone initially.
3. Train only CoSER-specific adapter / FiLM / control branch first.
4. Compare x_aux-only vs diffusion-refined x_hat.
5. Report LPIPS/DISTS improvements and semantic drift checks.
```

Stage 4 expected output:

```text
x_hat
diffusion runtime
adapter ablation
x_aux vs x_hat comparison
no-detail vs detail-conditioned diffusion comparison
```

---

## 5.5 Actual Bitstream and Evaluation

CoSER files:

```text
src/coserdic/entropy/bitstream.py
src/coserdic/entropy/semantic_tokens.py
scripts/eval_stage1_semantic_bitstream.py
scripts/eval_compressai_anchor.py
future: scripts/eval_cod_lite_baseline.py
future: scripts/eval_rdvq_baseline.py
future: scripts/eval_glc_baseline.py
future: scripts/eval_stablecodec_baseline.py
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

Valid paper-facing evaluation path:

```text
x
  -> compress()
  -> actual bitstream bytes
  -> decompress()
  -> x_hat
  -> metrics
```

Forbidden paper-facing shortcut:

```text
x
  -> forward()
  -> likelihood-estimated bpp only
  -> metrics
```

---

# 6. Baseline runner requirements

Before claiming Stage 5 Core-MVP results, the following baseline runners must exist or be explicitly waived with reasons:

```text
scripts/eval_cod_lite_baseline.py
scripts/eval_rdvq_baseline.py
scripts/eval_compressai_anchor.py
scripts/eval_glc_baseline.py, if official setup is reproducible
scripts/eval_stablecodec_baseline.py, if official setup is reproducible
```

Recommended priority:

```text
Priority A:
  CoD-Lite baseline
  RDVQ baseline
  CompressAI anchors

Priority B:
  GLC baseline
  StableCodec baseline

Priority C:
  AEIC
  OneDC
  Control-GIC
  ResULIC
  RDEIC
```

For each baseline run, record:

```text
repository URL
commit hash
license
checkpoint name
checkpoint source
command used
dataset
image preprocessing
metric implementation
estimated bpp, if available
actual bpp, if available
known deviations from official evaluation
```

If a baseline cannot be reproduced, write a short waiver note explaining why.

---

# 7. Immediate implementation checklist

Before implementing each next stage:

```text
1. Read the mapped official files for that stage.
2. Add a short design note saying which implementation lessons are adopted.
3. State which parts are deliberately not copied.
4. Keep CoSER-specific data flow explicit.
5. Add actual compress/decompress or roundtrip tests.
6. Log reference repo commit hashes in experiment docs.
```

Next recommended work:

```text
1. Build a minimal semantic token prior for Stage 2.
2. Start from a simple CoSER-owned categorical prior.
3. Use RDVQ top-k/escape tensor-rANS as the target implementation reference
   for the actual entropy-coded version, not necessarily the first step.
4. Compare actual payload bpp against the Stage 1 fixed-bits baseline computed
   from codebook size, token grid, and image resolution.
5. Keep the current best Stage 1 checkpoint fixed while testing entropy coding.
6. In parallel, audit CoD-Lite checkpoint key compatibility for Stage 4.
7. Add or update baseline runners before Stage 5 claims.
```

Fixed-bits note:

```text
The fixed-bits baseline is configuration-dependent.

For a token grid Hs x Ws and codebook size K:

  fixed_bits = Hs * Ws * ceil(log2(K))
  fixed_bpp  = fixed_bits / (H * W)

For the current example setting:
  K = 8192
  Hs x Ws = 16 x 24
  H x W = 512 x 768

This gives approximately:
  fixed_bpp ≈ 0.01270

If codebook size, token grid, padding, or image resolution changes, recompute
this value. Do not hard-code 0.01270 bpp as a universal constant.
```

---

# 8. Final instruction to implementers

Do not treat “scratch implementation” as “ignore official implementations.”

The expected behavior is:

```text
read the official implementation
reproduce or smoke-test the official baseline where possible
learn the stable design choices
implement the CoSER-owned version
document deviations
evaluate using actual bitstreams
```

Before merging a major module, the implementer should be able to answer:

```text
1. Which official repositories were checked?
2. Which implementation details were adopted?
3. Which parts were deliberately not copied?
4. Why does the resulting module remain CoSER-owned?
5. Can the module be evaluated through compress/decompress?
6. Does decompression use only bitstream-derived information?
7. How is actual bpp computed?
```

If the answer is unclear, add a design-decision note before continuing.
