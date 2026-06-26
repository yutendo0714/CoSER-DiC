# Design Decisions and Rationale

この文書は、CoSER-DiCで採用する設計判断と、その理由を明記する。

---

## Decision 1: MVPではone-step diffusionを採用する

### 理由

multi-step diffusionは品質が高い可能性があるが、decoder latencyが大きく、圧縮codecとしての実用性を主張しにくい。近年のStableCodec、OneDC、CoD-Lite、AEICはone-stepまたは軽量decoder方向に進んでいる。

### 実装方針

- 最初はpretrained one-step diffusion decoderまたはcompression-oriented decoderを利用する。
- backboneをfreezeし、adapter/FiLM/conditioning branchから学習する。
- fallbackとしてauxiliary decoder + GAN refinementを一時的に使ってよいが、最終主結果はdiffusion版にする。

---

## Decision 2: Semantic token gridはH/32 × W/32から始める

### 理由

0.003〜0.05 bppではtoken数が多すぎるとrateが支配的になる。H/32はsemantic structureを保持しつつtoken数を抑えやすい。

例: 512×768の場合、16×24=384 tokens。

codebook_size=8192ならraw indexは13 bits/tokenであり、raw semantic bppは約0.0127 bpp。entropy codingでさらに下げる。

---

## Decision 3: Detail latentはsemantic tokenと同じ空間解像度にする

### 理由

実装とconditioningが簡単で、semantic/detail alignmentを取りやすい。

```text
semantic: H/32 × W/32 × 256
detail:   H/32 × W/32 × 64
```

Full版ではmulti-scale detail latentを試してよい。

---

## Decision 4: control mapはMVPでは送らない

### 理由

0.003〜0.05 bppではmask/control streamのoverheadが無視できない。decoderが再現できないencoder-side mapを無料で使うと不公平比較になる。

### 方針

- encoder-side importance mapはtraining guidanceだけに使う。
- decoder-side importance mapはsemantic tokensとauxiliary reconstructionからdeterministicに推定する。
- maskを送るならactual bppに含める。

---

## Decision 5: FIDだけで勝利を主張しない

### 理由

FIDはdistribution realismを測るが、個別画像のfaithfulnessを保証しない。生成圧縮では、FIDが良くても文字・顔・物体数が変わる可能性がある。

### 方針

主要指標を分ける。

```text
Rate: actual bpp
Distortion: PSNR, MS-SSIM
Perceptual fidelity: LPIPS, DISTS
Distribution realism: FID, KID
Semantic faithfulness: DINO/CLIP, segmentation consistency
Critical detail: OCR, face ID, edge F-score
Human: 2AFC visual pleasing and faithfulness
```

---

## Decision 6: Full autoregressive priorはMVPでは避ける

### 理由

rateは下がるがdecodeが遅くなり、one-step diffusionによる実用性主張と衝突する。まずはhyperprior + grouped contextで始める。

---

## Decision 7: VLM-DPOは主貢献にしすぎない

### 理由

VLM judgeに過適合すると、人間には不自然な出力を選ぶ可能性がある。また、主張が「semantic-residual representation」から「VLM alignment」へ散らばる。

### 方針

VLM-DPOはFull版のpost-training補助として扱い、MVPの中核には入れない。
