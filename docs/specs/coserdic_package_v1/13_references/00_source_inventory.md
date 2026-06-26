# Source Inventory and Role in CoSER-DiC

Access date: 2026-06-26

この文書は、CoSER-DiCパッケージ作成時に確認した主要論文・公開実装・設計参照をまとめる。利用時は必ず各リポジトリのlicense、checkpoint利用条件、dataset利用条件を再確認すること。

---

## Foundational theory and RD LIC

| Item | Role | URL |
|---|---|---|
| Ballé et al., scale hyperprior | LICのVAE + hyperprior基礎 | https://openreview.net/forum?id=rkcQFMZRb |
| Minnen et al., joint autoregressive + hierarchical priors | hyperpriorとAR priorの相補性 | https://arxiv.org/abs/1809.02736 |
| Cheng et al. 2020 | GMM likelihood + attention | https://github.com/ZhengxueCheng/Learned-Image-Compression-with-GMM-and-Attention |
| ELIC | efficient space-channel context | https://openaccess.thecvf.com/content/CVPR2022/html/He_ELIC_Efficient_Learned_Image_Compression_With_Unevenly_Grouped_Space-Channel_Contextual_CVPR_2022_paper.html |
| CompressAI | PyTorch library/eval platform | https://github.com/InterDigitalInc/CompressAI |
| TCM | Transformer-CNN mixed architecture | https://github.com/jmliu206/lic_tcm |
| MLIC++ | multi-reference entropy model | https://github.com/jiangweibeta/mlic |

---

## Generative / perceptual compression

| Item | Role | URL |
|---|---|---|
| Rate-Distortion-Perception | R-D-P理論的背景 | https://proceedings.mlr.press/v97/blau19a.html |
| HiFiC | canonical GAN perceptual codec | https://hific.github.io/ |
| MS-ILLM / NeuralCompression | statistical fidelity / ILLM discriminator | https://github.com/facebookresearch/NeuralCompression |
| EGIC | semantic segmentation-guided discriminator | https://arxiv.org/html/2309.03244v3 |

---

## VQ / generative latent coding

| Item | Role | URL |
|---|---|---|
| GLC | VQ-VAE generative latent transform coding | https://github.com/jzyustc/GLC |
| RDVQ | differentiable VQ for R-D optimization | https://github.com/CVL-UESTC/RDVQ |
| Control-GIC | variable-rate VQGAN compression | https://github.com/lianqi1008/Control-GIC |
| DLF | dual semantic/detail latent fusion | https://dlfcodec.github.io/ |

---

## Diffusion-based compression

| Item | Role | URL |
|---|---|---|
| PerCo | ultra-low bitrate diffusion + VQ/global description | https://openreview.net/forum?id=ktdETU9JBg |
| DiffC | Gaussian diffusion compression/progressive coding | https://arxiv.org/abs/2206.08889 |
| TACO | text-guided encoding | https://github.com/effl-lab/TACO |
| ResULIC | semantic residual + compression-aware diffusion | https://github.com/NJUVISION/ResULIC |
| RDEIC | relay residual diffusion extreme compression | https://github.com/huai-chang/RDEIC |
| StableCodec | one-step diffusion extreme compression | https://github.com/LuizScarlet/StableCodec |
| CoD / CoD-Lite / GenCodec | compression-native diffusion foundation and lightweight real-time codec | https://github.com/microsoft/GenCodec |
| OneDC | one-step diffusion + semantic distillation | https://github.com/onedc-codec/onedc |
| AEIC | shallow encoder ultra-low bitrate perceptual codec | https://github.com/LuizScarlet/AEIC |
| CADC | content-adaptive diffusion compression | https://arxiv.org/abs/2602.21591 |
| VLIC | VLM preference/DPO for compression | https://arxiv.org/abs/2512.15701 |
| MRIDC | region-adaptive diffusion | https://openaccess.thecvf.com/content/CVPR2025/html/Xu_Decouple_Distortion_from_Perception_Region_Adaptive_Diffusion_for_Extreme-low_Bitrate_CVPR_2025_paper.html |

---

## License and reproducibility caution

- GitHub公開実装はlicenseが異なる。研究利用・派生実装・checkpoint利用可否を必ず確認する。
- 論文値はmetric実装、crop、bpp定義、side information計上方法が異なることがある。
- main tableには、可能な限り同一pipelineで再評価した値を使う。
- actual bppが取得できない手法はmain claimの根拠にしない。
