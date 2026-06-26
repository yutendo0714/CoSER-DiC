# Priority Baseline Matrix

この表は、CoSER-DiC研究で比較・参照する既存手法を整理する。

---

## Direct generative / low-bitrate competitors

| Method | Type | Why important | Package role |
|---|---|---|---|
| GLC | VQ/generative latent coding | generative VQ latent空間でultra-low bitrateを扱う直接競合 | Priority A |
| RDVQ | differentiable VQ RD optimization | CoSER-DiCのsoft VQ設計に近い | Priority A |
| Control-GIC | variable-rate VQGAN generative compression | dynamic granularity / variable-rate比較 | Priority B |
| DLF | semantic/detail dual branch | CoSER-DiCに非常に近い問題設定 | Priority B/A |
| ResULIC | semantic residual + compression-aware diffusion | residual-guided ultra-low-rateの直接競合 | Priority A |
| RDEIC | relay residual diffusion | extreme diffusion compression比較 | Priority A |
| StableCodec | one-step diffusion + dual-branch coding | one-step diffusion competition | Priority A |
| CoD-Lite / GenCodec | compression-native one-step diffusion | decoder土台候補かつ実用性比較 | Priority A |
| OneDC | one-step diffusion + semantic distillation | one-step diffusionの強力baseline | Priority B |
| AEIC | shallow encoder ultra-low bitrate | encoding efficiency比較 | Priority B |

---

## Classical / standard codecs

| Method | Role |
|---|---|
| JPEG | legacy anchor |
| JPEG2000 | classical transform baseline |
| BPG | HEVC intra-based strong baseline |
| AVIF | modern deployment baseline |
| JPEG XL | modern image codec baseline |
| VVC/VTM | strong handcrafted coding anchor |
| ECM | optional latest standard-oriented anchor |

---

## RD-oriented LIC

| Method | Role |
|---|---|
| Ballé hyperprior | foundational learned codec |
| Minnen joint autoregressive + hyperprior | foundational entropy model baseline |
| Cheng2020 GMM+attention | strong CompressAI-era baseline |
| ELIC | efficient space-channel context |
| TCM | Transformer-CNN hybrid |
| MLIC++ | multi-reference entropy model |
| HPCM / MambaIC / GLIC | recent RD-oriented references, optional |

---

## Perceptual / generative references

| Method | Role |
|---|---|
| HiFiC | canonical GAN-based perceptual compression |
| MS-ILLM | GAN/perceptual statistical fidelity reference |
| EGIC | semantic segmentation-guided discriminator and D-P curve control |
| PerCo | ultra-low bitrate perfect realism diffusion reference |
| TACO | text-guided image compression reference |
| DiffC | diffusion-based progressive/noisy-pixel compression reference |
| CADC | content-adaptive diffusion design reference |
| VLIC | VLM preference/DPO alignment reference |
| MRIDC | region adaptive diffusion reference |
