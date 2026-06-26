# Core MVP Literature Update

Date: 2026-06-27 JST

This note keeps only literature signals that affect Core MVP-v0 decisions.
Broader full-version ideas remain archived.

## Primary Sources Rechecked

- GenCodec / CoD-Lite: https://github.com/microsoft/GenCodec
- CoD-Lite checkpoints: https://huggingface.co/zhaoyangjia/CoD_Lite
- RDVQ: https://github.com/CVL-UESTC/RDVQ
- GLC: https://github.com/jzyustc/GLC
- StableCodec: https://github.com/LuizScarlet/StableCodec
- CompressAI: https://github.com/InterDigitalInc/CompressAI
- Robust NIC: https://arxiv.org/abs/2112.08691
- Training-free robust LIC defense: https://arxiv.org/abs/2401.11902
- Backdoor attacks on deep image compression: https://arxiv.org/abs/2302.14677
- Privacy-shielded image compression: https://arxiv.org/abs/2506.15201
- LVLM-oriented image compression: https://arxiv.org/abs/2407.17060

## Core MVP Implications

1. CoD-Lite is the right first diffusion initialization candidate because it is
   compression-native and lightweight, but its decoder does not natively accept
   CoSER semantic/detail streams. The CoSER adapter must stay in `src/coserdic`.

2. RDVQ is highly relevant to rate-aware VQ learning, but using its pretrained
   tokenizer by default would weaken the novelty claim. CoSER semantic VQ stays
   scratch-trained in MVP-v0.

3. StableCodec is a strong one-step diffusion baseline and useful fallback
   reference, but it should not become the default backbone unless CoD-Lite
   integration fails.

4. Robustness/security/privacy papers reinforce the bitstream discipline:
   decoder-side information must be transmitted, provenance must be logged, and
   model robustness should become a post-MVP evaluation track.

5. Task-aware / LVLM-aware compression is strategically important but should not
   enter the optimization objective before the semantic/detail/diffusion
   separation has been demonstrated.

## Current Research Priority

Proceed with Stage 1 semantic VQ from scratch.

The next technical risk is codebook collapse. Track:

- perplexity
- used code count
- dead-code ratio
- reconstruction visual samples
- layout/color preservation

