# Core MVP Literature Update

Date: 2026-06-27 JST  
Scope: active literature signals for CoSER-DiC Core-MVP implementation.

This note keeps the broad LIC survey connected to concrete Core-MVP decisions.
Paper-facing CoSER-DiC results must use actual `compress/decompress` bpp, not
forward likelihood estimates alone.

## Active CoSER-DiC Position

CoSER-DiC should be developed as a new codec with explicit stream separation:

```text
semantic VQ tokens
  -> entropy-aware semantic token coding
detail residual latents
  -> semantic-conditioned residual entropy model
auxiliary reconstruction
  -> deterministic fidelity anchor
one/few-step diffusion
  -> stochastic texture and perceptual realism
actual bitstream
  -> official transmitted bpp
```

The closest neighbors are RDVQ, DLF, GLC, StableCodec, OneDC, CoD/CoD-Lite,
ResULIC/RDEIC, AEIC, and recent task-aware/robust LIC work. CoSER-DiC should
not be framed as "just adding diffusion"; the defensible novelty is the
entropy-constrained semantic/detail/stochastic separation with actual bitstream
accounting.

## Primary Sources Rechecked

### Official implementations / baseline anchors

```text
CompressAI: https://github.com/InterDigitalInc/CompressAI
RDVQ: https://github.com/CVL-UESTC/RDVQ
GenCodec / CoD / CoD-Lite: https://github.com/microsoft/GenCodec
CoD-Lite checkpoints: https://huggingface.co/zhaoyangjia/CoD_Lite
GLC: https://github.com/jzyustc/GLC
StableCodec: https://github.com/LuizScarlet/StableCodec
OneDC: https://github.com/onedc-codec/onedc
DLF: https://github.com/dlfcodec/Dual-generative-Latent-Fusion
MambaIC: https://github.com/AuroraZengfh/MambaIC
HPCM: https://github.com/lyq133/LIC-HPCM
AEIC: https://github.com/LuizScarlet/AEIC
Bridging compressed image latents and MLLMs: https://github.com/NYCU-MAPL/BridgingCompressionMLLM
```

### Papers / project pages most relevant to Core-MVP

```text
RDVQ: https://arxiv.org/abs/2604.10546
CoD-Lite: https://arxiv.org/abs/2604.12525
CoD: https://arxiv.org/abs/2511.18706
StableCodec: https://arxiv.org/abs/2506.21977
OneDC: https://arxiv.org/abs/2505.16687
DLF: https://dlfcodec.github.io/
Dictionary entropy model: https://openaccess.thecvf.com/content/CVPR2025/html/Lu_Learned_Image_Compression_with_Dictionary-based_Entropy_Model_CVPR_2025_paper.html
HPCM: https://arxiv.org/abs/2507.19125
MambaIC: https://openaccess.thecvf.com/content/CVPR2025/html/Zeng_MambaIC_State_Space_Models_for_High-Performance_Learned_Image_Compression_CVPR_2025_paper.html
MRIDC: https://openaccess.thecvf.com/content/CVPR2025/html/Xu_Decouple_Distortion_from_Perception_Region_Adaptive_Diffusion_for_Extreme-low_Bitrate_CVPR_2025_paper.html
Robust NIC: https://arxiv.org/abs/2112.08691
Training-free robust LIC defense: https://arxiv.org/abs/2401.11902
Attack/defense analysis: https://arxiv.org/abs/2401.10345
Bitstream collisions: https://arxiv.org/abs/2503.19817
Backdoor attacks: https://arxiv.org/abs/2302.14677
Privacy-shielded image compression: https://arxiv.org/abs/2506.15201
LVLM-oriented image compression: https://arxiv.org/abs/2407.17060
VLA-oriented image compression / SPARC: https://arxiv.org/abs/2606.16253
```

## Representative Method Map

| Area | Representative methods | Main idea / novelty | Datasets and metrics | Public code | CoSER-DiC implication |
|---|---|---|---|---|---|
| RD LIC foundations | Ballé hyperprior, Minnen joint AR+hyperprior, Cheng2020, CompressAI anchors | End-to-end transform coding with hyperprior, autoregressive context, Gaussian/mixture likelihoods | Kodak, Tecnick, CLIC, DIV2K; PSNR, MS-SSIM, estimated and actual bpp | CompressAI model zoo | Use as RD anchors and entropy API reference; do not optimize CoSER for PSNR-only claims. |
| Strong entropy modeling | ELIC, MLIC/MLIC++, HPCM, Dictionary Cross-Attention Entropy, MambaIC | More expressive context models: channel/spatial grouping, multi-reference, progressive context, dictionary priors, SSMs | Kodak/CLIC/Tecnick; BD-rate, runtime, model complexity | HPCM/MambaIC public; dictionary paper public | Stage 2 should start simple, then borrow grouped/progressive ideas only if token entropy dominates. |
| VQ-based compression | RDVQ, GLC, Control-GIC, DLF | VQ/generative latents; RDVQ adds differentiable rate feedback to codebook assignments; DLF splits semantic/detail generative latents | Kodak, DIV2K-val, CLIC; LPIPS, DISTS, FID/KID, bpp | RDVQ, GLC, DLF public | Stage 1 stays CoSER-owned; adopt diagnostics and entropy-aware lessons, not pretrained tokenizers by default. |
| Low-bitrate generative compression | MS-ILLM, PerCo, DiffEIC/RDEIC, ResULIC, MRIDC | Optimize rate-perception; use GAN/diffusion priors, semantic residuals, region-adaptive maps | Kodak, DIV2K, CLIC; LPIPS, DISTS, FID, KID, PSNR guardrails | ResULIC/DiffEIC style code public; MRIDC paper | CoSER should report perceptual plus faithfulness metrics and explicitly transmit any side maps. |
| One/few-step diffusion codecs | StableCodec, OneDC, CoD-Lite, DiffO/SODEC, AEIC | Reduce diffusion latency using one-step/few-step decoders, compression-oriented pretraining, distillation, shallow/asymmetric encoders | Kodak/DIV2K/CLIC; FID/KID/DISTS, LPIPS, bpp, runtime/FPS | StableCodec, OneDC, GenCodec, AEIC public | CoD-Lite is preferred Stage 4 initialization; StableCodec/OneDC are baselines/references, not replacements for CoSER streams. |
| Task/downstream-aware compression | LVLM-oriented compression, Bridging compressed latents and MLLMs, SPARC for VLA, human-machine coding | Allocate bits by semantic/task relevance rather than pixel fidelity; sometimes decode to features or latents consumed by LVLM/VLA models | VQA/caption/MLLM tasks, robot control benchmarks, rate-accuracy/success | Some public repos | Post-MVP evaluation track; add semantic drift, OCR/face/detail subsets before task objective enters training. |
| Robustness/security/privacy | Robust NIC, training-free defense, attack/defense analysis, bitstream collisions, frequency-trigger backdoors, PSIC | LIC is vulnerable to source attacks, bpp inflation, exact bitstream collisions, backdoors, VLP privacy leakage | Attack PSNR/bpp/RD-cost, downstream face/segmentation, VLP privacy metrics | Mixed; mostly papers | Add robustness/security eval after Core-MVP; log provenance and ensure decoder uses only transmitted information. |

## Core-MVP Design Decisions From Literature

### Stage 1 semantic VQ

Decision:

```text
Keep scratch-trained CoSER semantic VQ.
Use RDVQ/GLC/CoD-Lite as L1-L2 references from Stage 1 onward.
Do not initialize the default semantic tokenizer from RDVQ.
```

Why:

```text
RDVQ shows entropy-aware VQ is important, but replacing CoSER Stage 1 with RDVQ
would weaken the semantic/detail/stochastic separation claim. The current
Stage 1 should instead log RDVQ-style VQ/entropy diagnostics and remain a
CoSER-owned semantic token stream.
```

Current required Stage 1 diagnostics:

```text
semantic-only PSNR/L1/MS-SSIM
active_codes_global
global_code_entropy_bits
assignment_sample_entropy_bits
assignment_avg_entropy_bits
soft_usage_entropy_bits
commitment/codebook MSE
actual fixed_bits and learned-token-codec bpp
token roundtrip
```

### Stage 2 semantic token entropy

Decision:

```text
Start with a simple CoSER-owned categorical token prior.
Keep fixed_bits as an audit baseline.
Use RDVQ top-k/escape tensor-rANS as the target implementation reference for
the practical entropy-coded version.
```

Reasoning:

```text
HPCM/MambaIC/dictionary entropy models show context can become very powerful,
but a heavy masked Transformer/SSM prior at Stage 2 would slow debugging and
obscure whether the semantic stream itself is useful. First prove actual
payload savings over 0.01270 bpp on the current 256x256 Kodak token grid.
```

### Stage 3 detail residual

Decision:

```text
Use semantic-conditioned residual coding with a simple Gaussian/hyperprior
start. The detail stream must not duplicate semantic tokens or become a full
pixel transform codec that dominates total bitrate.
```

Reference lessons:

```text
GLC and StableCodec both show that latent/residual coding plus generative
decoding is effective at ultra-low bpp. DLF is the closest semantic/detail
competitor, so CoSER must report semantic/detail bpp decomposition and
semantic-only/no-detail ablations clearly.
```

### Stage 4 diffusion reconstruction

Decision:

```text
Use CoD-Lite pretrained weights only as a diffusion backbone initialization
candidate, behind a CoSER conditioning adapter. Start by freezing most of the
backbone and training only the adapter/control path.
```

Reference lessons:

```text
CoD-Lite gives the strongest compression-native real-time diffusion signal.
StableCodec and OneDC confirm one-step diffusion is viable, but their native
latents are not CoSER semantic/detail streams.
```

### Robustness and privacy

Decision:

```text
Post-MVP, add robustness/security/privacy evaluation before strong deployment
claims. Do not mix these objectives into Core-MVP training until semantic,
detail, diffusion, and actual bitstream paths are stable.
```

Minimum future tests:

```text
clean vs adversarial source perturbations
bpp inflation under attack
semantic-token collision or collapse checks
bitstream corruption/roundtrip checks
frequency-trigger/backdoor sensitivity
privacy/semantic-redaction ablations for VLP models
```

## Current Research Priority

1. Finish Stage 1 freeze gate with diagnostic-aware Kodak analysis and actual
   bitstream evaluation.
2. Build Stage 2 semantic token prior with actual byte-stream evaluation.
3. Compare learned semantic-token payload bpp against current fixed_bits
   payload bpp.
4. Keep CoD-Lite checkpoint compatibility audit running in parallel, but do
   not start Stage 4 integration until Stage 2/3 stream interfaces are clean.

