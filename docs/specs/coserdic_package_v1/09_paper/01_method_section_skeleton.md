# Method Section Skeleton

## 3.1 Problem formulation

Define image `x`, bitstream `B`, reconstruction `x_hat`, total rate `R`, distortion `D`, perceptual objective `P`.

```math
min_θ E_x [ R(B) + λ_D D(x,x_hat) + λ_P P(x,x_hat) ]
```

Then decompose bitstream:

```math
B = B_s ∪ B_d ∪ B_h
```

where `B_s` is semantic token stream, `B_d` is detail residual stream, and `B_h` contains hyperprior/header.

---

## 3.2 Entropy-constrained semantic VQ

Describe:

```text
Semantic Encoder
Differentiable VQ
Soft assignment for rate gradients
Categorical entropy model
Semantic auxiliary decoder
```

Show equations for soft VQ and `R_s`.

---

## 3.3 Semantic-conditioned detail residual coding

Describe:

```text
Residual input construction
Detail residual encoder
Gaussian hyperprior
Semantic-conditioned mean/scale prediction
Residual rate R_d
```

Explain why semantic conditioning avoids redundant coding.

---

## 3.4 Auxiliary reconstruction as fidelity anchor

Explain `x_aux` and why it prevents diffusion hallucination.

---

## 3.5 Compression-oriented diffusion reconstruction

Describe one-step diffusion decoder conditioned on:

```text
x_aux
s_hat
d_hat
rate embedding
```

Emphasize decoder role: synthesize texture, not invent semantics.

---

## 3.6 Training objective

Present stage-wise training and final loss.

```math
L = R_s + R_d + λ_DD + λ_PP + λ_auxL_aux + λ_semL_sem + λ_advL_adv
```

Optional preference loss in appendix or Full method section.

---

## 3.7 Bitstream and actual bpp

Describe actual arithmetic coding and exact bpp accounting.

```math
bpp = 8 * bytes(B) / (H*W)
```

State that no original-derived side information is passed free to decoder.
