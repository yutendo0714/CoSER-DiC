# Human and VLM Preference Study

Full版で導入するhuman/VLM評価の仕様。

---

## 2AFC human study

同一originalに対して、2つの復元画像A/Bを提示する。

質問は2種類に分ける。

### Q1: Visual pleasing

```text
Which reconstruction looks more visually pleasing and natural?
```

### Q2: Faithfulness

```text
Which reconstruction is more faithful to the original image?
```

日本語実験なら:

```text
どちらの復元画像がより自然で見た目が良いですか？
どちらの復元画像が元画像により忠実ですか？
```

---

## なぜ質問を分けるか

生成圧縮は、自然に見えるが内容が変わる画像を生成する可能性がある。visual pleasingとfaithfulnessを混ぜると、hallucinationを見逃す。

---

## VLM-assisted evaluation

VLMを使う場合、以下を分ける。

```text
visual preference judge
faithfulness judge
text preservation judge
face/object consistency judge
```

VLM判定をそのまま真実とせず、human studyの補助にする。

---

## Hard negatives

DPOまたはranking lossに使う候補。

```text
residual branch off
text-like region residual dropped
face region residual dropped
semantic token randomly masked
different diffusion seed
over-smoothed reconstruction
over-generated reconstruction
```

---

## VLM prompt templates

### Visual preference

```text
You are judging two reconstructed images produced by image compression systems.
Which image looks more natural, visually pleasing, and free of artifacts?
Ignore tiny pixel-level differences unless they affect perceived quality.
Return only A, B, or Tie, and briefly state the reason.
```

### Faithfulness

```text
You are judging two reconstructed images against the original image.
Which reconstruction is more faithful to the original content, including object identity, layout, text, faces, small objects, and boundaries?
Penalize hallucinated or missing content even if the image looks natural.
Return only A, B, or Tie, and briefly state the reason.
```

### Text/face critical

```text
Focus on text, signs, faces, logos, and small objects.
Which reconstruction preserves these critical details more accurately compared with the original?
Return only A, B, or Tie, and briefly state the reason.
```

---

## Reporting

```text
visual preference win rate
faithfulness preference win rate
VLM-human agreement if human labels exist
per-dataset breakdown
per-bpp breakdown
failure examples where VLM disagrees with humans
```
