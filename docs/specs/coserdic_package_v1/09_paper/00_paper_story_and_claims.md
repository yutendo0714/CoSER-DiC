# Paper Story and Claims

## Title candidates

1. **CoSER-DiC: Compression-Oriented Semantic-Residual Diffusion for Extreme Image Compression**
2. **Entropy-Constrained Semantic-Residual Diffusion Codec for Ultra-Low-Rate Image Compression**
3. **Separating Semantics, Residuals, and Texture Priors for Ultra-Low-Rate Generative Image Compression**

---

## Abstract skeleton

Ultra-low-rate learned image compression must decide which information should be transmitted and which can be synthesized by a generative prior. Existing transform codecs preserve structure but produce blurry reconstructions, while generative codecs improve realism but may hallucinate details. We propose CoSER-DiC, an entropy-constrained semantic-residual diffusion codec that explicitly separates semantic structure, detail-critical residuals, and stochastic texture generation. CoSER-DiC transmits RD-aware semantic VQ tokens and semantic-conditioned residual latents, constructs an auxiliary reconstruction as a fidelity anchor, and uses a compression-oriented one-step diffusion decoder for realistic texture synthesis. Experiments at 0.003–0.05 bpp on Kodak, CLIC, and DIV2K show improved rate-distortion-perception trade-offs and better semantic faithfulness over strong generative compression baselines.

---

## Main claim

```text
Extreme-low-bitrate generative image compression benefits from explicitly separating semantic structure, detail-critical residuals, and stochastic texture generation under entropy-constrained coding.
```

---

## Contributions

### Contribution 1

We introduce an entropy-constrained semantic VQ branch that transmits rate-aware semantic tokens for global layout and object structure.

### Contribution 2

We propose a semantic-conditioned detail residual branch that selectively encodes hallucination-sensitive local information such as text, faces, boundaries, and small objects.

### Contribution 3

We condition a compression-oriented one-step diffusion decoder on auxiliary reconstruction and dual representations, enabling realistic texture synthesis while preserving fidelity anchors.

### Contribution 4, optional

We introduce hallucination-aware preference tuning and critical-detail evaluation to measure and reduce generative compression failures.

---

## What not to claim

```text
Do not claim universal PSNR SOTA.
Do not claim perfect reconstruction.
Do not claim no hallucination.
Do not claim fair comparison if side information is not counted.
Do not overclaim VLM-DPO as human-equivalent.
```

---

## Reviewer concerns and answers

### Concern: This is just a combination of existing modules.

Answer:

The contribution is not the use of VQ or diffusion alone, but the entropy-constrained separation of semantic and residual information, with different entropy models and decoder roles. Ablation demonstrates that each branch handles a different failure mode.

### Concern: Generative compression changes content.

Answer:

We explicitly measure OCR, face identity, segmentation consistency, DINO/CLIP similarity, and human faithfulness preference. Detail residual coding and auxiliary reconstruction are designed to reduce hallucination.

### Concern: bpp accounting is unfair.

Answer:

All reported results use actual arithmetic-coded bitstreams, including header, hyperprior, residual, and optional control streams. No original-derived side information is passed to the decoder without counting bits.

### Concern: Diffusion decoder is too slow.

Answer:

MVP uses one-step/few-step compression-oriented diffusion and reports arithmetic coding time, decode time, parameters, memory, and diffusion steps.
