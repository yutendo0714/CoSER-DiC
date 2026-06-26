# Rate Control Protocol

## MVP rate points

MVPでは以下を優先する。

```text
0.01 bpp
0.03 bpp
0.05 bpp
```

理由:

```text
0.003 / 0.005 bppは非常に難しく、MVPの安定性確認には厳しすぎる。
まず0.01〜0.05 bppでdual representationの有効性を確認する。
```

MVP安定後に追加:

```text
0.003 bpp
0.005 bpp
0.02 bpp
```

---

## Lagrangian training

最初はtarget-rate penaltyではなくLagrangian方式を推奨する。

```math
L = R + λD
```

またはR-D-Pの場合:

```math
L = R + λ_D D + λ_P P
```

複数λでcheckpointを作り、RD/RDP curveを引く。

---

## Target-rate conditioning

Full版ではtarget bppを条件として入れる。

```math
x_hat = D(s_hat, d_hat, rate_condition)
```

rate embeddingを以下に入れる。

```text
semantic entropy model
detail entropy model
quantization scale predictor
diffusion decoder FiLM
```

---

## bpp matching

baseline比較では完全同一bppに合わせるのが難しいため、複数rate pointを評価し、curve比較とBD-rateを用いる。

近傍比較の許容範囲:

```text
target bpp ±5%以内を優先
±10%を超える場合は同一rate比較として扱わない
```

---

## Branch bpp allocation

各checkpointで以下を必ず記録する。

```text
semantic_token_bpp
semantic_hyper_bpp
detail_latent_bpp
detail_hyper_bpp
header_bpp
control_bpp
actual_total_bpp
```

想定される健全な傾向:

```text
ultra-low bpp: semantic branch比率が高い
higher low bpp: detail branch比率が増える
hyper/header overheadが支配的でない
```

異常:

```text
detail branchが全rateでtotal bppの70%以上
semantic branchがほぼ使われない
header/control overheadが大きすぎる
```
