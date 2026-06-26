# Stage-wise Training Protocol

CoSER-DiCは一気にend-to-endで学習しない。必ずstage-wiseに進める。

---

## Stage 0: Baseline and environment setup

目的:

```text
評価pipeline、dataset、metric、actual bpp計算、baseline実行環境を固定する。
```

成果物:

```text
CompressAI baseline evaluation
GLC/RDVQ/ResULIC/StableCodec/CoD-Lite等の環境構築メモ
Kodak/CLIC/DIV2K metric script
run logging template
```

---

## Stage 1: Semantic VQ tokenizer pretraining

目的:

```text
semantic tokenだけで大域構造を保持する。
codebook collapseを防ぐ。
```

学習対象:

```text
E_s
Q_VQ
D_aux_s
```

loss:

```text
L1 + MS-SSIM + LPIPS + VQ commitment + codebook usage
```

成功条件:

```text
x_semが大域構造を保持している
codebook perplexityが極端に低くない
dead codeが支配的でない
```

---

## Stage 2: Semantic entropy model training

目的:

```text
VQ tokenをactual entropy codingできるようにする。
soft VQにrate feedbackを入れる。
```

学習対象:

```text
E_s
Q_VQ
semantic entropy model
D_aux_s
```

loss:

```text
R_s + auxiliary reconstruction + commitment
```

注意:

rateを強くしすぎるとsemantic tokenが情報を失う。弱すぎるとbppが高くなる。

---

## Stage 3: Detail residual branch training

目的:

```text
semantic tokenだけでは失われるcritical detailを補う。
residual branchがbitを使いすぎないようにする。
```

学習対象:

```text
E_d
Q_d
detail hyperprior
detail entropy model
D_aux
```

最初はsemantic branchをfreezeし、その後joint fine-tuningする。

成功条件:

```text
semantic-onlyよりLPIPS/DISTSが改善する
detail bppが支配的になりすぎない
文字・顔・細部のvisual fidelityが改善する
```

---

## Stage 4: Diffusion decoder adaptation

目的:

```text
x_auxを自然でperceptualに良い復元へ変換する。
texture realismを向上させる。
semantic driftを抑える。
```

学習対象:

```text
diffusion adapter
FiLM layers
conditioning blocks
必要ならdecoder一部
```

最初はdiffusion backboneをfreezeし、adapter / FiLM / condition branchのみ更新する。

成功条件:

```text
x_auxよりLPIPS/DISTS/FIDが改善する
PSNRが極端に落ちない
semantic layoutが変わらない
文字・顔の破綻が増えない
```

---

## Stage 5: End-to-end joint fine-tuning

目的:

```text
semantic branch、residual branch、diffusion decoderを共同最適化する。
rate-distortion-perception balanceを整える。
```

学習対象:

```text
all modules
```

ただし、初期段階では以下を低learning rateにする。

```text
VQ codebook
diffusion backbone
entropy model
```

---

## Stage 6: Variable-rate training

目的:

```text
1つのモデルで複数bppに対応する。
```

rate condition:

```text
rate_level ∈ {0.003, 0.005, 0.01, 0.02, 0.03, 0.05}
```

MVP後に導入する。

---

## Stage 7: Preference / hallucination-aware post-training

目的:

```text
自然だが内容が変わる復元を減らす。
human preferenceに寄せる。
FIDだけが良くfaithfulnessが悪い失敗を防ぐ。
```

導入:

```text
VLM pairwise judge
hard-negative generation
DPO / ranking loss
semantic consistency filter
```

Full版で導入する。
