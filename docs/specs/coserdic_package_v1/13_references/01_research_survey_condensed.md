# Condensed Survey for CoSER-DiC

## 1. RD-oriented LIC

Balléのscale hyperprior以降、LICはVAE型transform codingとentropy modelingを中心に発展した。Minnenらのjoint autoregressive + hierarchical priorは、hyperpriorとautoregressive contextが相補的であることを示した。Cheng2020はdiscretized Gaussian mixture likelihoodとattentionを導入し、entropy modelの柔軟性を高めた。ELIC、TCM、MLIC++は、space/channel context、Transformer-CNN hybrid、multi-reference entropy modelingによりRD性能と速度のバランスを改善してきた。

CoSER-DiCで使う教訓:

```text
entropy modelはactual bppに直結する。
full ARはrateを下げるがdecode速度を悪化させる。
MVPではhyperprior + grouped contextで始める。
```

---

## 2. Perceptual / GAN-based compression

HiFiCはGANをlearned compressionへ組み込み、人間評価で強いperceptual compressionを示した。MS-ILLMはimplicit local likelihood model discriminatorでstatistical fidelityを改善した。EGICはsemantic segmentation-guided discriminatorとoutput residual predictionによりdistortion-perception curve制御を行う。

CoSER-DiCで使う教訓:

```text
perceptual realismとdistortion fidelityは衝突する。
semantic guidanceやresidual anchoringはhallucination抑制に重要。
```

---

## 3. VQ-based generative compression

GLCはgenerative VQ-VAE latent空間でtransform codingを行い、pixel-space codingよりperception-alignedな表現を使う。RDVQはVQのhard assignment問題をsoft distributionで緩和し、rate lossをencoder/codebookへ伝える。Control-GICはVQGANのvariable-length codeやdynamic granularityでvariable-rate adaptationを行う。DLFはsemantic/detail branchの分解により、個別objectのdetail fidelityを補強する。

CoSER-DiCで使う教訓:

```text
VQ tokenはsemantic structureの伝送に向く。
しかしVQだけではdetailが落ちる。
RD-awareなVQ最適化が必要。
semantic/detail分離は有望だが、actual entropy codingとhallucination評価が必要。
```

---

## 4. Diffusion-based compression

PerCoは0.003 bpp級まで扱うdiffusion-based perceptual compressionを示したが、iterative diffusionは遅い。StableCodec、OneDC、CoD-Lite、AEICはone-stepまたは軽量diffusion方向に進み、実用速度を重視している。ResULICはsemantic residualをdiffusion generationへ組み込み、CADCはcontent-adaptive quantizationやauxiliary decoder-guided information concentrationを提案した。VLICはVLM preference/DPOで人間評価に近いcompressionを狙う。

CoSER-DiCで使う教訓:

```text
diffusion decoderはtexture realismに強いが、semantic faithfulnessを保証しない。
semantic/detail情報をbitstreamで送り、diffusionには生成補完を担わせる。
one-step/few-stepで速度を確保する。
```

---

## 5. CoSER-DiCの差分

CoSER-DiCは、GLC/RDVQのRD-aware VQ、DLF/ResULICのsemantic-detail/residual思想、StableCodec/OneDC/CoD-Liteのone-step diffusion、CADCのcontent-adaptive発想、VLICのpreference alignmentを参照する。

ただし、主張は「全部足した」ではない。

主張は以下である。

```text
極低bppでは、semantic structure、detail-critical residual、stochastic textureを分けてentropy制約下で最適化する必要がある。
```
