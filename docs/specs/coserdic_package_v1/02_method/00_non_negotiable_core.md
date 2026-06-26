# Non-negotiable Core of CoSER-DiC

この文書は、CoSER-DiCとして成立するために変更してはいけない中核を定義する。

---

## 変更してはいけない中核

CoSER-DiCの中核は以下である。

```text
1. Semantic VQ token stream
2. Detail residual latent stream
3. Semantic-conditioned residual entropy model
4. Auxiliary reconstruction
5. Compression-oriented one-step or few-step diffusion reconstruction
6. Actual entropy coding and actual bpp reporting
```

これらを外したものはCoSER-DiCとは呼ばない。

---

## CoSER-DiCではないもの

以下はCoSER-DiCの主張から外れる。

```text
単一continuous latent + diffusion decoderのみ
VQ index codingのみでresidual branchなし
estimated bppだけのforward-only評価
decoderがoriginal-derived captionやsegmentationを無料で使う手法
full autoregressive priorだけを工夫したRD-oriented codec
PSNR最適化だけを主目的にしたcodec
```

---

## 研究上の主張

悪い主張:

> 既存のVQ、diffusion、residual coding、VLM-DPOを全部組み合わせた。

良い主張:

> 極低bpp生成圧縮では、semantic information、detail-critical information、stochastic textureを単一latentに混在させることが非効率である。CoSER-DiCは、RD-aware semantic VQ tokenとsemantic-conditioned residual latentを分離して符号化し、compression-oriented diffusion decoderに条件づけることで、realismとfaithfulnessの両方を改善する。

---

## Contributionの固定

論文では、以下の3つを主貢献として扱う。

### Contribution 1: Entropy-constrained Semantic VQ

大域構造・semantic layoutをVQ tokenに分離し、soft VQによりrate lossをencoder/codebookへ伝える。

### Contribution 2: Semantic-conditioned Detail Residual Coding

semantic tokenでは保持できないcritical detailを、semantic-conditioned entropy modelで符号化し、hallucination-sensitive領域のfaithfulnessを補強する。

### Contribution 3: Compression-oriented Diffusion Reconstruction

auxiliary reconstructionとsemantic/detail latentを条件にしたone-step/few-step diffusion decoderにより、極低bppでtexture realismを生成する。

VLM-DPO、OCR loss、face identity loss、adaptive quantizationは重要な拡張だが、MVP論文の主貢献として詰め込みすぎない。
