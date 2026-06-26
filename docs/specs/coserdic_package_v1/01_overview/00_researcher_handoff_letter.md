# 研究者向け伝達文

本パッケージは、極低ビットレート画像圧縮のための新規手法 **CoSER-DiC: Compression-Oriented Semantic-Residual Diffusion Codec** をゼロから実装し、国際会議投稿を目指して研究を進めるための仕様書である。

目的は、0.003〜0.05 bppのultra-low / extreme-low bitrate領域において、既存のgenerative learned image compression手法より優れたRate-Distortion-Perception性能を達成することである。

本手法の中心仮説は、極低bppでは画像情報を単一latentに圧縮するのではなく、semantic structure、detail-critical residual、stochastic textureを分離して扱う方が効率的である、というものである。

具体的には、以下の3要素を統合する。

## 1. RD-aware Semantic VQ Token Coding

画像の大域構造・物体配置・semantic layoutをVQ tokenとして符号化する。VQ tokenはRDVQ型のsoft index distributionによりentropy lossを反映させ、rate-awareに学習する。

## 2. Semantic-conditioned Detail Residual Coding

semantic tokenだけでは保持できない顔、文字、小物、境界、identityなどのcritical detailを、semantic tokenに条件づけたresidual latentとして符号化する。Residual branchはGaussian / mixture hyperpriorによりentropy codingする。

## 3. Compression-oriented One-step Diffusion Decoding

semantic tokenとdetail residualから得られるauxiliary reconstructionを条件として、one-stepまたはfew-step diffusion decoderにより自然な高周波・textureを生成する。Diffusion decoderはCoD-Lite、StableCodec、OneDC、AEICなどの公開実装・pretrained modelを初期化候補としてよい。

---

# 最初に実装するもの

最初に実装すべきものはMVP v0である。

MVP v0では、以下を必ず実装する。

```text
Semantic Encoder
Differentiable VQ
Semantic categorical hyperprior
Semantic auxiliary decoder
Detail residual encoder
Detail Gaussian hyperprior
Semantic-conditioned residual prior
Joint auxiliary decoder
One-step diffusion decoder adapter
Actual bitstream coder
Kodak / CLIC / DIV2K evaluation
```

MVP v0では、以下は入れない。

```text
VLM-DPO
OCR loss
face identity loss
text conditioning
transmitted segmentation map
transmitted caption
complex Mamba entropy model
full autoregressive token prior
multi-step diffusion with many sampling steps
large-scale human study
```

理由は、研究の核がぼやけるからである。

---

# 厳守事項

- estimated bppではなく、actual arithmetic-coded bitstreamに基づくactual bppを必ず報告する。
- decoderはbitstream以外のoriginal-derived informationにアクセスしてはならない。
- caption、segmentation map、importance map、OCR maskなどを使う場合は、必ずbitstreamに含め、bppに計上すること。
- 主張の中心は「diffusionを使ったcodec」ではなく、**entropy-constrained semantic-residual representation** である。
- 成功例だけではなく、failure caseを保存・分析すること。

---

# 最初の成功条件

最初の成功条件は、0.01 / 0.03 / 0.05 bpp付近でKodak、CLIC、DIV2Kに対して評価でき、semantic-only、no-residual、no-diffusionのアブレーションを示せることである。

その後、0.003 / 0.005 bpp、adaptive quantization、VLM preference tuning、critical-detail evaluationへ拡張する。

---

# 論文投稿に向けた最終要求

論文投稿に向けては、LPIPS、DISTS、FID、KIDだけでなく、OCR、face identity、semantic consistency、human 2AFC、runtime、actual bpp decompositionを報告し、CoSER-DiCの各moduleがどのように貢献しているかを明確に示すこと。
