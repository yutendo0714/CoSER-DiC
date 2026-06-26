# Risk Register and Debugging Guide

## Risk 1: VQ codebook collapse

症状:

```text
codebook perplexityが低い
同じtokenばかり使う
x_semが単調
```

対策順:

```text
1. commitment lossを下げる
2. EMA codebook updateを確認
3. usage regularizationを強める
4. codebook sizeを一時的に4096へ下げる
5. semantic decoderの容量を少し上げる
6. temperature annealingを遅くする
```

---

## Risk 2: Detail branchが全部送る

症状:

```text
detail bppが高すぎる
semantic branchが使われない
x_auxが普通のautoencoder復元に近い
```

対策順:

```text
1. lambda_R_dを上げる
2. detail channel数を64から32へ下げる
3. semantic branchをfreezeしてdetailだけが逃げないようにする
4. detail latent sparsity regularizationを入れる
5. residual入力からxそのものを外し、error中心にする
```

---

## Risk 3: Diffusion decoderがhallucinateする

症状:

```text
見た目は良いが文字・顔・物体が変わる
x_auxよりfaithfulnessが落ちる
```

対策順:

```text
1. x_aux conditioningを強める
2. diffusion adapterのみ学習し、backbone更新を抑える
3. L1 / semantic consistency lossを上げる
4. detail branchのcritical情報を増やす
5. hard negative preference tuningを追加する
```

---

## Risk 4: LPIPSは良いがDISTSやsemantic metricsが悪い

症状:

```text
perceptualには近いが構造がずれる
```

対策:

```text
1. semantic lossを上げる
2. DINO feature lossを追加
3. segmentation consistency評価を導入
4. diffusion decoderの自由度を下げる
```

---

## Risk 5: actual bppが高すぎる

症状:

```text
estimated bppよりactual bppが高い
target bppに届かない
```

対策:

```text
1. entropy model calibrationを確認
2. CDF precisionを確認
3. arithmetic coderのoverheadを分解
4. header overheadを確認
5. semantic/detail/hyper別bppを分析
6. token priorを強化
```

---

## Risk 6: baselineに勝てない

診断:

```text
どのmetricで負けているかを分解する。
LPIPS/DISTSで負ける -> perceptual decoder不足
FID/KIDで負ける -> diffusion prior/adv loss不足
OCR/faceで負ける -> residual branch不足
actual bppで負ける -> entropy model不足
runtimeで負ける -> AR prior/diffusion implementationが重い
```

対策:

```text
1. ablationで弱点moduleを特定
2. direct competitorを1つに絞って改善
3. 0.01〜0.05 bppに焦点を戻す
4. full拡張を増やす前にMVPのrate allocationを改善
```
