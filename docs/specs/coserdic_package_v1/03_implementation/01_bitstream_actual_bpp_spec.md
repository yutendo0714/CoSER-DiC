# Bitstream and Actual bpp Specification

low-bitrate生成圧縮では、side informationの扱いが研究の信頼性を左右する。この仕様は必ず守る。

---

## bpp定義

actual bppは以下で計算する。

```math
bpp = 8 * total_bitstream_bytes / (H * W)
```

含めるもの:

```text
header
semantic token stream
semantic hyper stream
detail latent stream
detail hyper stream
optional control stream
checksum if stored
```

含めないもの:

```text
model weights
decoder weights
metric logs
debug tensors
```

ただし、decoderへ外部情報を渡すなら必ずbitstreamに入れる。

---

## Header schema

最低限、以下を含める。

```text
magic_number
codec_version
image_height
image_width
padded_height
padded_width
color_space
rate_level_id
perception_level_id
semantic_token_grid_shape
detail_latent_shape
entropy_model_version
cdf_precision
stream_lengths
checksum
optional_seed
```

---

## Stream order

推奨順序:

```text
[HEADER]
[SEMANTIC_HYPER_STREAM]
[SEMANTIC_TOKEN_STREAM]
[DETAIL_HYPER_STREAM]
[DETAIL_LATENT_STREAM]
[OPTIONAL_CONTROL_STREAM]
```

---

## Semantic stream

```text
semantic_hyper_stream:
  optional hyperprior symbols for semantic token probability

semantic_token_stream:
  arithmetic-coded VQ token indices
```

semantic tokenはcategorical entropy modelで符号化する。

---

## Detail stream

```text
detail_hyper_stream:
  hyperlatent for residual Gaussian / logistic distribution

detail_latent_stream:
  arithmetic-coded residual latent symbols
```

---

## Control stream policy

MVPではcontrol streamを送らない。

理由:

```text
0.003〜0.05 bppではcontrol map overheadが大きい。
original-derived maskを無料で送ると不公平比較になる。
```

Full版でmaskを送る場合:

```text
binary maskまたはlow-resolution mapとしてentropy codingする。
必ずactual bppに含める。
maskあり/なしのablationを行う。
```

---

## Decoder leakage禁止

decoderは以下を使ってはいけない。

```text
original image
original imageから得たcaption
original imageから得たsegmentation map
original imageから得たOCR mask
original imageから得たface mask
original imageから得たimportance map
training-only debug tensor
```

これらを使う場合は、bitstreamに含める。

---

## Estimated vs actual bpp report

各評価で以下を保存する。

```yaml
estimated_bpp_total:
actual_bpp_total:
semantic_bpp:
semantic_hyper_bpp:
detail_bpp:
detail_hyper_bpp:
header_bpp:
control_bpp:
difference_ratio: (actual - estimated) / estimated
```

差分目安:

```text
10%以内: 許容。ただし記録する。
10〜20%: 要調査。
20%以上: entropy coding/probability calibrationに問題あり。
```

---

## Round-trip test

必須test:

```text
x -> compress -> bitstream -> decompress -> x_hat
```

確認:

```text
encoded semantic tokens == decoded semantic tokens
encoded detail latents == decoded detail latents
同一bitstreamから同一x_hat
padding後に元サイズへ正しくcrop
```
