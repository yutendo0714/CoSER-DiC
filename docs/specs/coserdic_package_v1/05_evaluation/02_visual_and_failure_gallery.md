# Visual Comparison and Failure Gallery Plan

生成圧縮ではvisual evidenceが極めて重要である。ただし成功例だけでは不十分で、failure caseも必ず示す。

---

## Standard comparison layout

各図は以下を並べる。

```text
Original
BPG / VVC
RD-oriented LIC
GLC / RDVQ
StableCodec / ResULIC / CoD-Lite
CoSER-DiC
```

必要に応じてbppを各画像下に表示する。

---

## Crop visualization

以下を拡大する。

```text
face
text
small object
thin structure
background texture
object boundary
repetitive texture
natural stochastic texture
```

---

## Failure case categories

必ず以下の失敗例を収集する。

```text
文字が崩れる
顔identityが変わる
物体数が変わる
細部が過生成される
textureが不自然になる
semantic tokenが誤る
背景が自然だが構造が変わる
x_auxはfaithfulだがx_hatでhallucinationする
```

---

## Visual ablation

同一画像・同一bpp付近で以下を比較する。

```text
semantic-only
semantic+detail
semantic+diffusion
semantic+detail+diffusion
```

目的:

```text
semantic branchが構造を保持すること
detail branchがcritical detailを補うこと
diffusion decoderがtexture realismを改善すること
```

---

## Bit allocation visualization

以下を可視化する。

```text
semantic token usage map
detail residual energy map
estimated detail bpp map
uncertainty/importance map if available
x_sem vs x_aux vs x_hat difference map
```

---

## 禁止

```text
成功例だけを選ぶ
bppが大きく違う画像を同一比較に見せる
estimated bppだけを表示する
original-derived side informationを使った結果を通常結果に混ぜる
```
