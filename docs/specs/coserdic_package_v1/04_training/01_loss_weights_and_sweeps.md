# Loss Weights and Sweep Plan

この文書は初期loss weightと探索範囲を定義する。最終値ではなく、安定実装の出発点である。

---

## Stage 1: Semantic VQ pretraining

```yaml
loss:
  l1_sem: 1.0
  ms_ssim_sem: 0.2
  lpips_sem: 0.1
  vq_commitment: 0.25
  codebook_usage: 0.01
```

観察:

```text
x_semの構造保持
codebook perplexity
dead code ratio
semantic-only LPIPS
```

調整:

```text
x_semが崩れる -> l1/ms_ssimを上げる
textureに寄りすぎる -> semantic decoder容量を下げる、LPIPSを下げる
codebook collapse -> usage regularizationを上げる、temperature annealingを遅くする
```

---

## Stage 2: Semantic entropy training

```yaml
loss:
  rate_semantic: sweep [0.25, 0.5, 1.0, 2.0, 4.0]
  l1_sem: 1.0
  lpips_sem: 0.1
  vq_commitment: 0.25
```

調整:

```text
semantic bppが高すぎる -> rate_semanticを上げる
layoutが崩れる -> rate_semanticを下げる、l1/ms_ssimを上げる
actual bppがestimatedより高い -> entropy calibrationを確認
```

---

## Stage 3: Detail residual training

```yaml
loss:
  rate_semantic: sweep [0.5, 1.0, 2.0]
  rate_detail: sweep [1.0, 2.0, 4.0, 8.0]
  l1_aux: 1.0
  ms_ssim_aux: 0.2
  lpips_aux: 0.2
  dists_aux: 0.1
```

調整:

```text
detail branchが全部送る -> rate_detailを上げる、detail channelsを減らす
critical detailが改善しない -> rate_detailを下げる、semantic-conditioned priorを確認
x_auxがぼやける -> lpips/distsを上げる
```

---

## Stage 4: Diffusion decoder adaptation

```yaml
loss:
  rate_total: 1.0
  l1_hat: 0.5
  lpips_hat: 0.3
  dists_hat: 0.2
  adversarial: 0.05
  semantic_consistency: 0.1
  l1_aux_anchor: 0.5
```

調整:

```text
FIDは改善したがfaithfulnessが悪化 -> l1_hat / semantic_consistency / aux_anchorを上げる
textureが弱い -> lpips/dists/adversarialを上げる
PSNRが落ちすぎる -> l1_hatとaux_anchorを上げる
```

---

## Stage 5: Joint fine-tuning

```yaml
learning_rates:
  semantic_encoder: 1.0e-5_to_5.0e-5
  vq_codebook: 1.0e-6_to_1.0e-5
  entropy_models: 1.0e-5_to_5.0e-5
  detail_encoder: 1.0e-5_to_1.0e-4
  aux_decoder: 1.0e-5_to_1.0e-4
  diffusion_adapter: 1.0e-5
  diffusion_backbone: 0_or_1.0e-6
```

---

## Sweepの優先順位

最初から全lossをgrid searchしない。

推奨順:

```text
1. rate_semantic / rate_detail
2. detail channels: 32 vs 64
3. codebook size: 4096 vs 8192
4. diffusion conditioning strength
5. perceptual/adversarial weights
6. adaptive quantization parameters
```
