# MVP v0 Reference Architecture

この文書は、研究者が迷わないように最初に実装すべき固定構成を定義する。

---

## 全体フロー

```text
Input image x
  ↓
Semantic Encoder E_s
  ↓
Differentiable VQ Q_VQ
  ↓
Semantic categorical entropy model
  ↓
Semantic tokens s_hat
  ↓
Semantic auxiliary reconstruction x_sem
  ↓
Detail residual encoder E_d(x, x_sem, residual)
  ↓
Detail Gaussian hyperprior entropy model conditioned on s_hat
  ↓
Detail residual latent d_hat
  ↓
Joint auxiliary decoder D_aux(s_hat, d_hat)
  ↓
Auxiliary reconstruction x_aux
  ↓
One-step diffusion decoder G_diff(x_aux, s_hat, d_hat, rate condition)
  ↓
Final reconstruction x_hat
```

---

## 入力・前処理

```yaml
input:
  color_space: RGB
  value_range: [0, 1]
  training_crop_size: 256
  later_finetune_crop_size: 512
  eval: full_resolution
  pad_to_multiple: 64
  remove_padding_before_metric: true

augmentation:
  random_crop: true
  horizontal_flip: true
  color_jitter: false_initially
  random_resize: false_initially
```

評価時はcenter cropしない。Kodak / CLIC / DIV2K validationはfull-resolutionで評価する。paddingした領域はmetric計算から除外する。

---

## Semantic Encoder

初期reference構成:

```text
Input: x ∈ R^{3×H×W}

Conv 3×3, C=128, stride=2
ResDownBlock C=128, stride=2
ResDownBlock C=192, stride=2
ResDownBlock C=256, stride=2
ResDownBlock C=256, stride=2
ResBlock C=256 × 4
Conv 1×1, C=256

Output:
h_s ∈ R^{256×H/32×W/32}
```

activation: `SiLU`  
normalization: `GroupNorm`

---

## Differentiable VQ

```yaml
vq:
  codebook_size: 8192
  embedding_dim: 256
  use_ema_update: true
  commitment_weight: 0.25
  temperature_init: 1.0
  temperature_final: 0.1
  straight_through: true
  soft_rate_estimation: true
```

学習時はsoft assignmentでrateを推定し、forward reconstructionではhard indexを使う。


a. soft assignment:

```math
q(k=j|h_s)=softmax(-||h_s-e_j||^2 / τ)
```

b. hard forward:

```math
k^* = argmin_j ||h_s-e_j||^2
```

c. semantic rate:

```math
R_s = E_{q(k|h_s)}[-log_2 p_θ(k|C_s)]
```

監視:

```text
codebook perplexity
dead code ratio
average token entropy
semantic bpp
```

---

## Semantic Categorical Entropy Model

単純に8192 channelのlogitを全token位置に出すと重くなるため、MVPでは低ランクlogit形式を推奨する。

各token位置 `i` について、context embedding `c_i` を予測する。

```math
c_i = f_θ(z_s, local_context)
```

codebook embedding `e_j` とcontext embeddingの内積でlogitを作る。

```math
ℓ_{i,j} = e_j^T W c_i + b_j
p(k_i=j|C_i)=softmax(ℓ_{i,j})
```

MVP構成:

```yaml
semantic_entropy:
  hyperprior: true
  grouped_context: true
  full_autoregressive: false
  low_rank_logits: true
  cdf_precision: 16
```

---

## Semantic Auxiliary Decoder

semantic tokenだけから粗い画像を復元する。

```text
Input:
  quantized semantic embedding s_hat

Architecture:
  ResBlock C=256 × 4
  UpBlock C=256 -> 192
  UpBlock C=192 -> 128
  UpBlock C=128 -> 96
  UpBlock C=96 -> 64
  UpBlock C=64 -> 32
  Conv 3×3 -> RGB

Output:
  x_sem ∈ R^{3×H×W}
```

目的:

```text
semantic tokenがlayoutと構造を持つようにする。
diffusion decoderに丸投げしない。
```

---

## Detail Residual Encoder

入力:

```text
concat[x, x_sem, x - x_sem, |x - x_sem|]
```

channel数:

```text
3 + 3 + 3 + 3 = 12 channels
```

構成:

```text
Conv 3×3, C=128, stride=2
ResDownBlock C=128, stride=2
ResDownBlock C=192, stride=2
ResDownBlock C=256, stride=2
ResDownBlock C=256, stride=2
Semantic FiLM conditioning
ResBlock C=256 × 4
Conv 1×1, C=64

Output:
h_d ∈ R^{64×H/32×W/32}
```

---

## Detail Residual Quantization

MVPでは標準的なscalar quantizationを使う。

Training:

```text
add uniform noise U(-0.5, 0.5)
```

Inference:

```text
round
```

Full版でadaptive quantizationへ拡張する。

---

## Detail Entropy Model

MVPではGaussian conditionalを使う。

```math
p(d_hat_i | z_d, s_hat) = N(μ_i, σ_i^2)
```

semantic-conditioned mean/scaleを使う。

```text
semantic_context = SemanticContextNet(s_hat)
hyper_context = HyperDecoder(z_d)
concat_context = concat[semantic_context, hyper_context]
mu, sigma = ParamNet(concat_context)
```

```yaml
detail_entropy:
  distribution: gaussian
  hyperprior: true
  semantic_conditioned: true
  mixture_components: 1_initially
  min_scale: 0.11
```

---

## Joint Auxiliary Decoder

semantic embeddingとdetail latentを結合して補助復元を作る。

```text
Input:
  s_hat ∈ R^{256×H/32×W/32}
  d_hat ∈ R^{64×H/32×W/32}

concat -> Conv 1×1 C=256
ResBlock × 6
Upsample × 5
RGB output
```

出力: `x_aux`

役割:

```text
distortion anchor
diffusion condition
semantic/detail alignment
```

---

## Diffusion Decoder

MVPではone-step diffusion decoderを基本とする。

入力:

```text
x_aux
s_hat
d_hat
rate_level_embedding
```

conditioning:

```text
x_aux: image condition
s_hat, d_hat: multi-scale FiLM / adapter condition
rate_level: embedding -> FiLM
```

基本方針:

```text
pretrained compression-oriented diffusion decoderを利用可能なら初期化に使う。
最初はbackboneをfreezeし、adapter / FiLM / condition branchのみ学習する。
```

Fallback:

```text
pretrained diffusion decoderがうまく使えない場合、
auxiliary decoder + GAN/perceptual refinement generatorを暫定decoderとして実装する。
ただし最終論文ではdiffusion decoder版を主結果にする。
```
