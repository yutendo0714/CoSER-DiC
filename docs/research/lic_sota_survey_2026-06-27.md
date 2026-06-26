# LIC / Generative Compression / Robust Compression Survey

Updated: 2026-06-27 JST

This survey is the working map for CoSER-DiC. It intentionally separates
published claims from values that must be re-measured in our pipeline. Main
tables for the paper should use actual compress/decompress bpp whenever a
method exposes practical bitstreams.

## CoSER-DiC Positioning

CoSER-DiC should be framed as an entropy-constrained semantic-residual
representation for ultra-low-bitrate generative compression:

- semantic VQ stream: layout, object identity, global structure
- detail residual stream: faces, text, small objects, boundaries, local fidelity
- auxiliary reconstruction: deterministic fidelity anchor
- one/few-step diffusion: stochastic texture and high-frequency realism
- actual bitstream: header, semantic stream, residual stream, hyper streams

The closest direct neighbors are DLF, ResULIC, StableCodec, RDVQ, OneDC, and
CoD-Lite. The strongest novelty claim should not be "we add diffusion"; it
should be the explicit semantic/detail/stochastic separation under entropy
constraints and actual bitstream accounting.

## RD-Oriented Learned Image Compression

Representative line:

- Ballé scale hyperprior: VAE-style transform coding plus hyperprior, foundation
  for end-to-end LIC.
- Minnen joint autoregressive + hierarchical prior: combines hyperprior and AR
  context to improve entropy estimates.
- Cheng2020: discretized Gaussian mixture likelihood plus attention.
- ELIC / MLIC / MLIC++ / HPCM: stronger context models and multi-reference or
  progressive context modeling.
- TCM / MambaIC / GLIC-like models: Transformer, CNN, and state-space model
  hybrids for RD optimization.
- CompressAI: practical PyTorch platform with pretrained models and actual
  entropy-coded evaluation utilities.

CoSER-DiC implication:

- use CompressAI as the first RD baseline and entropy-coding reference;
- avoid full pixel-wise AR in the MVP because decode speed matters;
- semantic and residual priors can start with hyperprior + grouped context;
- later, borrow progressive/channel-context ideas for residual entropy only if
  residual bits dominate or estimated/actual bpp calibration is poor.

Primary source notes:

- CompressAI provides pretrained LIC models and evaluation scripts for learned
  and classical codecs: https://github.com/InterDigitalInc/CompressAI
- HPCM official code is public: https://github.com/lyq133/LIC-HPCM

## VQ-Based and Generative Latent Compression

### GLC

GLC performs transform coding in a generative VQ-VAE latent space rather than
pixel space. The official README states that the latent space offers stronger
sparsity, semantics, and perception alignment, and reports ultra-low image
rates below 0.04 bpp on CLIC-style evaluation. It also introduces spatial
categorical hyper modules and code-prediction loss for semantic consistency.

- Official implementation: https://github.com/jzyustc/GLC
- Dataset/metrics: Kodak, CLIC, perceptual metrics including FID/DISTS/LPIPS.
- CoSER-DiC delta: GLC is mostly generative-latent token coding; CoSER-DiC
  keeps a separate residual stream and diffusion reconstruction anchor.

### RDVQ

RDVQ directly targets the mismatch between VQ representation learning and
entropy modeling by using a differentiable relaxation of codebook distributions.
Its abstract reports CVPR 2026 Oral acceptance and bitrate savings versus
RDEIC on DIV2K-val under DISTS/LPIPS comparisons.

- Paper: https://arxiv.org/abs/2604.10546
- Code: https://github.com/CVL-UESTC/RDVQ
- CoSER-DiC delta: adopt the differentiable rate feedback principle, but use the
  VQ stream specifically as the semantic stream rather than the full codec.

### Control-GIC

Control-GIC uses a VQGAN framework with variable-length codes and dynamic
granularity adaptation. It is important for variable-rate control and dynamic
patch granularity.

- Paper/code: https://arxiv.org/abs/2406.00758 and
  https://github.com/lianqi1008/Control-GIC
- CoSER-DiC delta: variable-rate semantic token allocation is a later extension;
  MVP should keep fixed grid tokens to preserve ablation clarity.

### DLF

DLF decomposes generative latents into semantic and detail elements with two
branches and cross-branch interaction. It reports strong quality even below
0.01 bpp and bitrate savings against MS-ILLM on CLIC2020.

- Project/code: https://dlfcodec.github.io/ and
  https://github.com/dlfcodec/Dual-generative-Latent-Fusion
- CoSER-DiC delta: DLF is the closest semantic/detail competitor. CoSER-DiC
  must be stricter about actual entropy-coded bpp decomposition and diffusion
  anchoring via auxiliary reconstruction.

## Diffusion and Perceptual Compression

### PerCo / DiffC / TACO

These methods establish that diffusion priors can push perceptual quality at
extreme rates, including global descriptions, noisy-pixel/progressive coding,
and text-guided coding. They are important references, but multi-step sampling
and/or text side-information can complicate fair actual-bpp accounting.

CoSER-DiC implication:

- avoid free text/caption/segmentation side information;
- if a control stream is added later, entropy-code it and report control bpp;
- prefer one-step/few-step diffusion for practical runtime.

### ResULIC

ResULIC combines semantic residual coding and compression-aware diffusion. It
has public code, BSD-3-Clause license, and model links for 0.03, 0.01, and
0.002 bpp variants. Its inference uses a few DDIM steps.

- Official implementation: https://github.com/NJUVISION/ResULIC
- Datasets: Kodak, DIV2K, CLIC-style evaluation.
- CoSER-DiC delta: keep residual branch, but make the residual entropy model
  explicitly semantic-conditioned and keep semantic tokens as a first-class
  entropy-coded stream.

### StableCodec

StableCodec is a one-step diffusion codec for extreme image compression. It
uses a deep compression latent codec plus a dual-branch coding structure and
reports strong FID/KID/DISTS down to around 0.005 bpp. The repository provides
training/inference code and checkpoints for several Kodak bpp points.

- Official implementation: https://github.com/LuizScarlet/StableCodec
- CoSER-DiC delta: StableCodec is a strong one-step diffusion baseline and a
  possible decoder-initialization reference. CoSER-DiC should not copy its
  latent-only framing; the semantic VQ/detail residual split remains central.

### CoD / CoD-Lite / GenCodec

GenCodec hosts CoD, a compression-native diffusion foundation model, and
CoD-Lite, a real-time one-step diffusion codec. The README emphasizes high
realism/fidelity, extreme compression, real-time coding, and downstream
applications. CoD-Lite reports a lightweight encoder/decoder and high FPS on
A100-class hardware.

- Official implementation: https://github.com/microsoft/GenCodec
- CoSER-DiC delta: use CoD/CoD-Lite as a preferred diffusion backbone candidate
  if licensing/checkpoints fit, especially because it is compression-native.

### OneDC

OneDC argues that multi-step sampling is unnecessary for generative image
compression. It uses a latent compression module plus one-step diffusion and
semantic distillation from a pretrained generative tokenizer to the hyperprior.

- Project/code: https://onedc-codec.github.io/ and
  https://github.com/onedc-codec/onedc
- CoSER-DiC delta: semantic distillation into the entropy side information is
  relevant, but CoSER-DiC should transmit explicit semantic VQ tokens and keep
  the detail residual branch separate.

### AEIC

AEIC explores shallow/moderate encoders for real-time ultra-low bitrate
perceptual compression. It provides checkpoints, an entropy coding engine build
path, and practical bitstream support via `--use_practical_entropy_coding`.

- Official implementation: https://github.com/LuizScarlet/AEIC
- CoSER-DiC delta: useful for runtime and asymmetric encoder/decoder design.
  CoSER-DiC can later test shallow semantic/detail encoders after MVP accuracy
  is stable.

### MRIDC / CADC / VLIC

MRIDC introduces region-adaptive diffusion and map-guided latent masking for
extreme-low bitrate perception compression.

- Paper: https://openaccess.thecvf.com/content/CVPR2025/html/Xu_Decouple_Distortion_from_Perception_Region_Adaptive_Diffusion_for_Extreme-low_Bitrate_CVPR_2025_paper.html

CADC proposes uncertainty-guided adaptive quantization, auxiliary-decoder-guided
information concentration, and bitrate-free adaptive textual conditioning
derived from auxiliary reconstruction.

- Paper: https://arxiv.org/abs/2602.21591

VLIC uses VLM binary preference judgments for diffusion codec post-training and
reports human-aligned gains. It is a strong reference for later preference
tuning, not MVP.

- Paper: https://arxiv.org/abs/2512.15701

CoSER-DiC implication:

- adaptive quantization and region allocation are high-value extensions;
- any original-derived map must be transmitted or derived decoder-side;
- VLM/DPO preference tuning should be post-MVP and evaluated separately from
  semantic/detail entropy contributions.

## Low-Bitrate and Perceptual Metrics

Primary metric bundle for CoSER-DiC:

- actual bpp and bpp decomposition
- LPIPS-Alex, DISTS
- FID/KID on CLIC/DIV2K-sized sets, not Kodak-only
- PSNR-RGB, PSNR-Y, MS-SSIM as fidelity guardrails
- DINO/CLIP similarity for semantic drift checks
- OCR edit distance and face-ID similarity for critical-detail subsets
- encode/decode/arithmetic-coding runtime and peak memory

Avoid claiming PSNR SOTA. For generative compression, a fair claim needs both
perceptual quality and faithfulness checks.

## Robustness, Security, Privacy

### Adversarial robustness

"Towards Robust Neural Image Compression" shows that negligible adversarial
perturbations on source images can severely degrade decoded reconstructions and
studies preprocessing/self-ensemble and adversarial finetuning defenses.

- Paper: https://arxiv.org/abs/2112.08691

"A Training-Free Defense Framework for Robust Learned Image Compression" uses
random input transforms and two-way compression to improve robustness without
modifying existing codecs.

- Paper: https://arxiv.org/abs/2401.11902

CoSER-DiC implication:

- add a robustness eval mode after MVP: clean, PGD-like source attack, random
  transform defense, adversarial finetune option;
- separately report rate inflation and reconstruction degradation;
- check whether semantic tokens are more attack-sensitive than residual latents.

### Backdoor/security

"Backdoor Attacks Against Deep Image Compression via Adaptive Frequency Trigger"
demonstrates DCT-domain triggers that can attack compression quality and
downstream face/segmentation tasks.

- Paper: https://arxiv.org/abs/2302.14677

CoSER-DiC implication:

- checkpoint provenance and external pretrained weights need explicit logging;
- add trigger sensitivity tests for semantic token collapse and residual misuse.

### Privacy-aware compression

PSIC targets privacy against vision-language pretrained models by producing
bitstreams with multiple decoding options: perceptually satisfactory but
semantically shielded by default, or full semantic reconstruction under a
custom condition.

- Paper/code: https://arxiv.org/abs/2506.15201 and
  https://github.com/JiayinXu5499/PSIC

CoSER-DiC implication:

- privacy-aware variants are downstream extensions;
- the semantic/detail split is a natural control surface for privacy, but it
  must not leak original-derived information outside the bitstream.

## Task-Aware / Downstream-Aware Compression

LVLM-oriented image compression trains a variable-bitrate codec and pre-editing
module with semantic-token losses from large vision-language models, improving
rate-accuracy for multimodal tasks.

- Paper: https://arxiv.org/abs/2407.17060

S2-CoT studies parameter-efficient adaptation of codecs for machine vision and
argues that adapting the entropy model's statistical semantics matters, not only
the encoder/decoder feature structure.

- Paper/code: https://arxiv.org/abs/2604.10017 and
  https://github.com/Brock-bit4/S2-CoT

CoSER-DiC implication:

- downstream-aware compression should be an evaluation track, not the first
  optimization target;
- semantic residual latents may be valuable for OCR, face, and VLM tasks;
- entropy model adaptation is a good later route for task-specific fine-tuning.

## Recommended CoSER-DiC Research Path

1. Stage 0: fix environment, datasets, bitstream container, logging, baseline
   registry, and CompressAI actual-bpp evaluation.
2. Stage 1: train semantic VQ tokenizer with strong auxiliary reconstruction and
   monitor codebook perplexity/dead-code ratio.
3. Stage 2: add differentiable semantic rate feedback and actual token stream
   coding. Validate estimated/actual bpp gap.
4. Stage 3: add semantic-conditioned detail residual hyperprior. Keep residual
   bpp constrained so it does not become a standard transform codec.
5. Stage 4: attach a one-step diffusion decoder, preferably initialized from a
   compression-native checkpoint if feasible. Freeze the backbone first.
6. Stage 5: joint fine-tune with actual-bpp evaluation and ablations:
   semantic-only, no residual, no diffusion, no semantic-conditioned prior.
7. Extensions: variable-rate, adaptive quantization, critical-detail losses,
   VLM preference tuning, robustness/privacy/task-aware variants.

