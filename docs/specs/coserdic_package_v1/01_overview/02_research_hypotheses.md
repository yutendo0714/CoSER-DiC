# Research Hypotheses and Required Evidence

この文書は、CoSER-DiCで実証すべき仮説と、それぞれに必要な実験証拠を定義する。

---

## Hypothesis 1: RD-aware Semantic VQは通常VQよりrate-efficientである

### 主張

Semantic VQ tokenをRD-awareに学習すると、hard VQや通常のVQ token codingより低いactual bppで同等以上のsemantic fidelityを得られる。

### 検証実験

```text
hard VQ vs soft RDVQ-style VQ
factorized prior vs categorical hyperprior
estimated bpp vs actual bpp
codebook perplexity
semantic-only reconstruction quality
DINO / CLIP similarity
```

### 成功判定

- soft VQ + entropy feedbackでactual semantic bppが下がる。
- semantic-only reconstructionのlayoutが大きく崩れない。
- codebookがcollapseしていない。

---

## Hypothesis 2: Detail residual branchはhallucination-sensitive領域のfaithfulnessを改善する

### 主張

Semantic tokenだけでは失われるcritical detailを、semantic-conditioned residual latentとして符号化することで、文字、顔、小物、境界などのfaithfulnessが改善する。

### 検証実験

```text
semantic-only vs semantic+detail
residual without semantic condition vs semantic-conditioned residual
critical region crop comparison
OCR / face ID / edge F-score
semantic/detail bpp decomposition
```

### 成功判定

- detail branch追加でLPIPS/DISTSが改善する。
- OCRまたはface identityなどのcritical metricが改善する。
- detail branchがtotal bppを支配しすぎない。

---

## Hypothesis 3: Diffusion decoderはtexture realismを担わせるべきであり、semantic/detail情報はbitstreamで送るべきである

### 主張

Diffusion decoder単体にsemantic/detailを任せるとhallucinationが起こる。semantic tokenとdetail residualをbitstreamで送った上で、diffusion decoderにはtexture realismを担わせる方が、realismとfaithfulnessを両立しやすい。

### 検証実験

```text
aux-only
semantic+diffusion
semantic+detail+diffusion
generic diffusion prior vs compression-oriented prior
one-step vs few-step
```

### 成功判定

- diffusion追加でFID/KID/LPIPS/DISTSが改善する。
- PSNR、OCR、face ID、semantic consistencyが大きく崩れない。

---

## Hypothesis 4: Adaptive rate allocationはcritical detailのbit効率を改善する

### 主張

uniform quantizationでは、重要領域と背景textureに同じ符号化原理が適用されるため非効率である。decoder-sideでも再現可能なuncertainty/importanceに基づくadaptive quantizationにより、同じbppでcritical detailをより効率よく保持できる。

### 検証実験

```text
uniform quantization vs adaptive quantization
critical region LPIPS
text/face/object crop metrics
bit allocation heatmap
```

### 成功判定

- total bppを増やさずcritical region指標が改善する。
- 背景textureに過剰なbitが割かれていない。

---

## Hypothesis 5: VLM preference tuningはhard-negative設計と組み合わせたときにhallucination抑制へ効く

### 主張

VLM-DPOを単に「見た目の良さ」に使うと、faithfulnessを犠牲にする可能性がある。hard negativeを使い、faithfulness promptと組み合わせることでhallucination抑制にも使える。

### 検証実験

```text
no preference tuning
visual-only VLM preference
faithfulness-aware VLM preference
hard negativeあり/なし
human 2AFC: visual pleasing vs faithful to original
```

### 成功判定

- human 2AFCでvisual preferenceが改善する。
- faithfulness質問でも悪化しない、または改善する。
- OCR/face/semantic metricsが悪化しない。
