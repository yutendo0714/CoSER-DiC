# Acceptance Criteria

この文書は、各stageで「成功した」と判断する基準と、研究者からの成果物受け入れ基準を定義する。

---

## Stage 1成功条件: Semantic VQ

```text
x_semが大域構造を保持している
codebook perplexityが極端に低くない
dead codeが支配的でない
semantic token gridが妥当な構造を持つ
semantic-only reconstructionが完全崩壊していない
```

失敗:

```text
codebook usageが数十code程度にcollapse
x_semが平均画像に近い
layoutが保持されない
```

---

## Stage 2成功条件: Entropy model

```text
estimated semantic bppが下がる
actual semantic bppとestimated semantic bppの差が小さい
semantic reconstructionが大きく崩れない
```

目安:

```text
actual bpp / estimated bpp の差が10%以上なら要調査
20%以上ならentropy codingまたはprobability calibrationに問題あり
```

---

## Stage 3成功条件: Detail branch

```text
semantic-onlyよりLPIPS/DISTSが改善する
detail bppが支配的になりすぎない
文字・顔・細部のvisual fidelityが改善する
```

危険:

```text
0.03 bpp付近でdetail branchがtotal bppの70%以上を常に使う場合、普通のtransform codec化している可能性がある。
```

---

## Stage 4成功条件: Diffusion decoder

```text
x_auxよりLPIPS/DISTS/FIDが改善する
PSNRが極端に落ちない
semantic layoutが変わらない
文字・顔の破綻が増えない
```

危険:

```text
FIDは改善したがOCRやface identityが悪化する場合、diffusionがhallucinationしている。
```

---

## MVP完了条件

```text
[ ] compress/decompress APIが動作する
[ ] actual bitstreamから復元できる
[ ] estimated bppとactual bppを両方reportできる
[ ] 0.01 / 0.03 / 0.05 bpp付近で結果がある
[ ] Kodak / CLIC / DIV2Kで評価済み
[ ] semantic-only / no-residual / no-diffusion ablationがある
[ ] 最低2つ以上の直接競合baselineと比較済み
[ ] visual comparisonがある
[ ] failure case galleryがある
```

---

## 投稿候補と判断する条件

```text
[ ] LPIPSまたはDISTSで主要生成系baselineを上回る
[ ] FID/KIDが同等以上
[ ] actual bpp込みで比較している
[ ] semantic faithfulness指標で改善がある
[ ] human 2AFCまたはVLM-assisted評価で補強できる
[ ] ablationで各moduleの役割が示せる
[ ] runtimeがmulti-step diffusion系より明確に良い
```

PSNRでSOTAを取る必要はない。ただし、PSNRが極端に低い場合は「見た目だけcodec」と批判されるため、下限は維持する。

---

## 納品物チェックリスト

```text
[ ] CoSER-DiC source code
[ ] README
[ ] environment file
[ ] training configs
[ ] evaluation configs
[ ] pretrained checkpoints
[ ] compress script
[ ] decompress script
[ ] actual bitstream examples
[ ] Kodak evaluation results
[ ] CLIC / DIV2K evaluation results
[ ] baseline comparison table
[ ] ablation table
[ ] visual comparison figure
[ ] failure case gallery
[ ] estimated vs actual bpp report
[ ] semantic/detail bitrate decomposition
[ ] runtime benchmark
[ ] reproducibility notes
[ ] paper-ready method description
```

---

## 未完了扱い

以下は未完了扱いにする。

```text
estimated bppだけの結果
bitstreamなしのforward-only評価
baselineなしの自手法結果
visualだけでmetricなし
metricだけでfailure caseなし
decoderがoriginal-derived side informationを使っている結果
```
