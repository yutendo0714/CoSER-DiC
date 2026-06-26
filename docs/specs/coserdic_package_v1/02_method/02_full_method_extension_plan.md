# Full Method Extension Plan

MVPが動作し、baseline比較と主要ablationが成立した後に追加する拡張を定義する。

---

## Extension 1: Variable-rate Control

### 目的

1つのモデルで複数bppに対応する。

### rate levels

```text
0.003, 0.005, 0.01, 0.02, 0.03, 0.05
```

### conditioning方法

```text
rate_level -> embedding -> FiLM parameters
rate_level -> entropy model prior
rate_level -> quantization scale
rate_level -> diffusion decoder condition
```

### 注意

MVPでは0.01 / 0.03 / 0.05 bppのsingle-rateまたは少数rateモデルを優先し、variable-rateはFull版にする。

---

## Extension 2: Adaptive Quantization

### 目的

critical detailにbitを集中し、背景textureやstochastic detailはdiffusion priorに任せる。

### 制約

decoderでも再現可能な情報だけからquantization scaleを予測する。

```math
Δ_d(i,c) = f_Δ(s_hat, z_d, rate_condition)
```

### 禁止

encoder側だけが知るoriginal image由来のimportance mapを、無料でdecoderへ渡してはいけない。

使う場合はbitstreamに含め、bppに計上する。

---

## Extension 3: Cross-branch Interaction

semantic branchとdetail branchの冗長性を下げる。

候補:

```text
semantic -> detail entropy prior
detail -> diffusion adapter
x_aux -> decoder-side uncertainty map
semantic/detail cross-attention
residual latent sparsity gating
```

最初はsemantic-conditioned detail entropy modelだけで十分。Full版でcross-attentionを追加する。

---

## Extension 4: Hallucination-aware Preference Tuning

### 目的

自然だが内容が変わる復元を抑える。

### hard negatives

```text
residual branch off
text-like region residual dropped
face region residual dropped
semantic token randomly masked
different generation seed
over-smoothed reconstruction
over-generated reconstruction
```

### VLM prompts

- Visual preference prompt
- Faithfulness-to-original prompt
- Text/face/object preservation prompt

VLM-DPOは主lossではなく、post-trainingの補助alignmentとして使う。

---

## Extension 5: Critical-detail Evaluation and Optional Losses

最終論文でhallucination抑制を主張するため、以下を導入する。

```text
OCR accuracy
OCR edit distance
face identity similarity
segmentation consistency
object count consistency
edge F-score
critical region LPIPS / DISTS
```

最初は評価指標として導入し、必要に応じてloss化する。

---

## 拡張を入れる順序

```text
1. variable-rate control
2. adaptive quantization
3. critical-detail evaluation
4. VLM-DPO / preference tuning
5. advanced entropy model or cross-branch attention
```

MVPが安定する前に拡張を入れすぎると、どの部品が効いたか判断できなくなる。
