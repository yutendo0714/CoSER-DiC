# Limitations and Responsible Reporting

## Limitations to acknowledge

```text
生成圧縮はhallucinationを完全には避けられない。
低bppではPSNRがRD-oriented codecsより低くなる可能性がある。
OCR/face/semantic-critical領域では専用lossなしでは破綻が残る可能性がある。
one-step diffusionはmulti-step diffusionより生成品質で劣る可能性がある。
actual bppで比較すると論文値より弱く見える可能性がある。
```

---

## Responsible reporting

生成圧縮の結果は、自然に見えても内容が変化することがある。特に次の用途では注意が必要。

```text
医療画像
衛星・監視画像
法務・証拠画像
文書・OCR用途
顔認証・本人確認
商品画像・ロゴ
```

CoSER-DiCをこれらの用途へ適用する場合は、perceptual metricsではなくtask-specific faithfulness metricを主指標にする。

---

## Paperで明記すべきこと

```text
本手法は低bppでperceptual realismを高めるが、lossy generative compressionである。
復元画像は原画像と完全同一ではない。
critical-detail evaluationを行ったが、全用途でsemantic correctnessを保証するものではない。
```
