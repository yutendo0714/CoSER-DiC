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
PICO/practical perceptual LIC, ResULIC/RDEIC, AEIC, and recent
task-aware/robust LIC work. CoSER-DiC should
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
PICO / practical perceptual LIC: https://apple.github.io/ml-pico/
Diffusion bit-flip robustness: https://openaccess.thecvf.com/content/CVPR2026W/AIGENS/papers/Vaisman_On_the_Robustness_of_Diffusion-Based_Image_Compression_to_Bit-Flip_Errors_CVPRW_2026_paper.pdf
```

### 2026-06-27 web recheck notes

```text
GenCodec / CoD / CoD-Lite:
  The public GenCodec repo frames CoD as a CVPR 2026 compression-native
  diffusion foundation model and CoD-Lite as the real-time/lightweight
  derivative. This supports keeping CoD-Lite as Stage 4 backbone initialization
  and design reference, not as a replacement for CoSER semantic/detail streams.

AEIC:
  The public AEIC repo emphasizes ultra-low bitrate (<0.05 bpp), shallow
  encoders, and real-time encoding. This strengthens the case for keeping
  CoSER Stage 3/4 encoder complexity modest rather than building a large
  residual encoder too early.

MambaIC / DCAE:
  MambaIC confirms the current strong-entropy-model direction: SSM/context
  modeling plus local attention for high-resolution efficiency. DCAE confirms
  that dataset-level external priors can improve entropy estimation. For
  CoSER, these are Stage 2/3 reference ideas after the current top-k bridge and
  static residual anchors are stable.

SPARC:
  SPARC is a very recent task-aware LIC signal for VLA systems, using temporal
  latent masking and tilted rate loss for bitrate-success tradeoffs. This
  should remain post-MVP, but its tilted-rate lesson is relevant if CoSER
  later over-suppresses rare semantic/detail events.

PICO:
  Practical LIC work stresses perceptual quality, runtime, on-device latency,
  and subjective/user-facing quality, not just BD-rate. CoSER should add runtime
  and container overhead reporting once the streams stabilize.

RobustNIC and newer robustness/security papers:
  Robustness remains under-tested in LIC. Source perturbation, bpp inflation,
  bitstream collision, backdoor, privacy, and bit-flip checks should be
  explicit post-MVP evaluation axes.
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
| Practical perceptual LIC | PICO / practical learned codec studies | Balance perceptual quality, runtime, deployability, and bitstream constraints rather than only BD-rate | Runtime/FPS, perceptual metrics, actual coding constraints | Public project page | Once Stage 2/3 are stable, CoSER should report runtime and container overhead, not just rate-distortion/perception curves. |
| Robustness/security/privacy | Robust NIC, training-free defense, attack/defense analysis, bitstream collisions, frequency-trigger backdoors, diffusion bit-flip robustness, PSIC | LIC is vulnerable to source attacks, bpp inflation, exact bitstream collisions, backdoors, bit flips, VLP privacy leakage | Attack PSNR/bpp/RD-cost, downstream face/segmentation, bit-flip recovery, VLP privacy metrics | Mixed; mostly papers | Add robustness/security eval after Core-MVP; log provenance and ensure decoder uses only transmitted information. |

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
Start with a simple CoSER-owned categorical token prior and real-byte bridge.
Keep fixed_bits as an audit baseline.
Use RDVQ top-k/escape tensor-rANS as the target implementation reference for
the practical entropy-coded version.
```

Reasoning:

```text
HPCM/MambaIC/dictionary entropy models show context can become very powerful,
but a heavy masked Transformer/SSM prior at Stage 2 would slow debugging and
obscure whether the semantic stream itself is useful. First prove actual
payload savings over the active Stage 2 baseline, not just estimated
cross-entropy or fixed_bits.
```

Current actual-bpp gate:

```text
fixed_bits payload_bpp: 0.012695
previous left-context static Huffman: 0.011729
active learned top512/escape bridge: 0.010722
roundtrip: true with decoder-schedule top-k reconstruction

active prior:
  outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json

token prior:
  checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt
```

### Stage 3 detail residual

Decision:

```text
Use semantic-conditioned residual coding with a simple Gaussian/hyperprior
start. The detail stream must not duplicate semantic tokens or become a full
pixel transform codec that dominates total bitrate.
```

Current bootstrap gate:

```text
active Stage 2 semantic payload on Kodak: 0.010722 bpp
Stage 3 active uniform residual d32/b5/r0.25 position-Huffman detail payload on Kodak: 0.008250 bpp
total semantic+detail payload on Kodak: 0.018972 bpp
compact CRC32 short-ID full CoSERBitstream payload on Kodak: 0.031301 bpp
quality delta vs semantic-only: +0.3957 dB PSNR, +0.00576 MS-SSIM
roundtrip: true for semantic tokens and detail residual codes
low-rate d32/b4/r0.25 position-Huffman anchor: 0.016205 total payload bpp, 0.028534 compact CRC32 short-ID full-stream bpp, +0.3705 dB PSNR
active cross-dataset compact CRC32 short-ID full-stream bpp:
  Kodak: 0.031301, DIV2K100: 0.031302, CLIC valid: 0.030202
active cross-dataset PSNR delta:
  Kodak: +0.3957 dB, DIV2K100: +0.4601 dB, CLIC valid: +0.7171 dB
low-rate cross-dataset compact CRC32 short-ID full-stream bpp:
  Kodak: 0.028534, DIV2K100: 0.028591, CLIC valid: 0.027502
low-rate cross-dataset PSNR delta:
  Kodak: +0.3705 dB, DIV2K100: +0.3055 dB, CLIC valid: +0.5649 dB
previous zlib-fixed d32/b4/r0.25 detail payload: 0.008759 bpp
previous global-Huffman d32/b5/r0.25 total payload: 0.019007 bpp
previous JSON-header active full-stream payload: 0.080399 bpp
previous compact SHA256 active full-stream payload: 0.037282 bpp
previous compact CRC32 long-ID active full-stream payload: 0.033864 bpp
```

Learned residual AE probe:

```text
implemented semantic-conditioned residual AE with actual transmitted detail payload
2500step no-rate: 0.025370 total payload bpp, +0.2073 dB, +0.01159 MS-SSIM
1000step rate-proxy 0.03 from no-rate: 0.015066 total payload bpp, +0.0112 dB, +0.00007 MS-SSIM
decision: not active; keep as scaffold until a real entropy objective or
teacher-distilled residual target beats the static-Huffman anchors
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

1. Replace the fixed residual-grid detail path only when a learned residual
   entropy model beats the b5/b4 position-Huffman anchors under actual
   compact CoSERBitstream bytes.
2. Tighten Stage 2 real-byte coding toward the RDVQ top-k/escape tensor-rANS
   reference, with the current Huffman bridge kept as the active baseline.
3. Keep CoD-Lite checkpoint compatibility audit running in parallel, but do
   not start Stage 4 integration until Stage 2/3 stream interfaces are clean.
4. Add robustness/security evaluation after the semantic/detail streams are
   stable, especially bitstream corruption and source-attack bpp inflation.
