# Mathematical Formulation and Losses

## Problem formulation

入力画像を `x`、復元画像を `x_hat` とする。

CoSER-DiCはsemantic token `s_hat` とdetail latent `d_hat` を送る。

```math
R(s_hat, d_hat) = R_s(s_hat) + R_d(d_hat)
```

最終目的:

```math
min_θ E_x [ R_s + R_d + λ_D D(x, x_hat) + λ_P P(x, x_hat) ]
```

ここで、`D` はdistortion、`P` はperceptual/fidelity lossである。

---

## Semantic branch

```math
h_s = E_s(x)
```

soft VQ:

```math
q(k=j|h_s)=softmax(-||h_s-e_j||^2/τ)
```

hard VQ:

```math
k^*=argmin_j ||h_s-e_j||^2
s_hat=e_{k^*}
```

rate:

```math
R_s = E_{q(k|h_s)}[-log_2 p_θ(k|C_s)]
```

semantic reconstruction:

```math
x_sem = D_{aux,s}(s_hat)
```

---

## Detail branch

residual input:

```math
r_in = concat(x, x_sem, x-x_sem, |x-x_sem|)
```

```math
h_d = E_d(r_in, s_hat)
```

quantization:

```math
d_hat = round(h_d)
```

entropy model:

```math
p(d_hat_i | z_d, s_hat) = N(μ_i, σ_i^2)
```

rate:

```math
R_d = -Σ_i log_2 p(d_hat_i | z_d, s_hat)
```

---

## Auxiliary reconstruction

```math
x_aux = D_aux(s_hat, d_hat)
```

Auxiliary decoderは、diffusion decoderの前にfidelity anchorを作る。

---

## Diffusion reconstruction

one-step版:

```math
x_hat = G_diff(x_aux, s_hat, d_hat, c_rate)
```

few-step版:

```math
x_hat = D_φ(x_T, x_aux, s_hat, d_hat, c_rate)
```

MVPではone-stepを推奨する。

---

## Stage 1 loss: Semantic VQ pretraining

```math
L_S1 =
1.0 * ||x-x_sem||_1
+ 0.2 * (1-MS-SSIM(x,x_sem))
+ 0.1 * LPIPS(x,x_sem)
+ 0.25 * L_commit
+ 0.01 * L_usage
```

目的:

```text
semantic tokenだけで大域構造を復元する。
codebook collapseを防ぐ。
```

---

## Stage 2 loss: Semantic entropy training

```math
L_S2 = λ_Rs R_s
+ 1.0 * ||x-x_sem||_1
+ 0.1 * LPIPS(x,x_sem)
+ 0.25 * L_commit
```

初期探索:

```yaml
lambda_R_s: [0.25, 0.5, 1.0, 2.0, 4.0]
```

---

## Stage 3 loss: Detail residual training

```math
L_S3 = λ_Rs R_s + λ_Rd R_d
+ 1.0 * ||x-x_aux||_1
+ 0.2 * (1-MS-SSIM(x,x_aux))
+ 0.2 * LPIPS(x,x_aux)
+ 0.1 * DISTS(x,x_aux)
```

初期探索:

```yaml
lambda_R_s: [0.5, 1.0, 2.0]
lambda_R_d: [1.0, 2.0, 4.0, 8.0]
```

注意:

detail branchはsemantic branchより強くrate制約する。そうしないとdetail branchが普通のtransform codecのように全部を送ってしまう。

---

## Stage 4 loss: Diffusion decoder adaptation

```math
L_S4 = R_s + R_d
+ 0.5 * ||x-x_hat||_1
+ 0.3 * LPIPS(x,x_hat)
+ 0.2 * DISTS(x,x_hat)
+ 0.05 * L_adv
+ 0.1 * L_sem
+ 0.5 * ||x-x_aux||_1
```

注意:

`x_hat`だけを良くしようとするとdiffusion decoderがhallucinateする。`x_aux` lossを残してfidelity anchorを維持する。

---

## Stage 5 loss: Joint fine-tuning

```math
L_S5 = R_s + R_d
+ λ_D D(x,x_hat)
+ λ_P P(x,x_hat)
+ λ_aux L_aux
+ λ_sem L_sem
+ λ_adv L_adv
```

---

## Semantic consistency loss

```math
L_sem = ||F_DINO(x)-F_DINO(x_hat)||_1 + ||F_CLIP(x)-F_CLIP(x_hat)||_1
```

CLIP lossを強くしすぎると、原画像に忠実というより「CLIP的にそれらしい画像」へ寄る危険がある。DINO / segmentation featureの方が構造保持には使いやすい。

---

## GAN loss

MVPではoptional。使う場合はhinge GANを推奨する。

Generator:

```math
L_G = -E[D(x_hat)]
```

Discriminator:

```math
L_D = E[max(0,1-D(x))] + E[max(0,1+D(x_hat))]
```

---

## Preference / DPO loss

Full版で導入する。

```math
L_DPO = -log σ(β[(log p_φ(x_hat^+|c)) - (log p_φ(x_hat^-|c))])
```

VLM-DPOは主目的ではなく、hallucination抑制とhuman-aligned perceptual refinementの補助として使う。
