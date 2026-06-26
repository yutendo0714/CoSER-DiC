# Unit Tests and Sanity Checks

実装初期から以下のテストをCIまたは手動チェックに入れる。

---

## Test 1: Bitstream round-trip

```text
x -> compress -> bitstream -> decompress -> x_hat
```

確認:

```text
decompressはoriginal imageにアクセスしない
bitstreamだけから復元できる
同じbitstreamなら同じx_hatが出る
```

---

## Test 2: Entropy coding consistency

```text
encoded semantic tokens == decoded semantic tokens
encoded detail latents == decoded detail latents
```

1 symbolでも違えば失敗。

---

## Test 3: Estimated vs actual bpp

```text
estimated_bpp
actual_bpp
difference_ratio
```

差が大きい場合は必ず原因を記録する。

---

## Test 4: Padding test

任意サイズ画像で確認する。

```text
input arbitrary size
pad to multiple of 64
compress/decompress
crop to original size
compute metrics only on original region
```

---

## Test 5: Decoder leakage test

decoderが以下を使っていないことを確認する。

```text
original image
encoder-side-only mask
original-derived caption
original-derived segmentation
original-derived OCR map
original-derived face mask
debug tensor
```

---

## Test 6: Deterministic decompression

```text
same bitstream -> same x_hat
```

diffusion samplingがstochasticな場合、seedをheaderに含める。ただしMVPではdeterministic one-step decoderを推奨する。

---

## Test 7: Loss component sanity

各stageでlossを個別に0/有効化して、勾配が期待moduleに流れるか確認する。

```text
R_s -> semantic encoder / semantic entropy
R_d -> detail encoder / detail entropy
L_aux -> aux decoder / encoders
L_diff/perceptual -> diffusion adapter
```

---

## Test 8: Codebook usage sanity

監視:

```text
codebook perplexity
dead code ratio
top-k token frequency
semantic token entropy
```

異常:

```text
数十codeしか使わない
特定tokenが支配的
x_semがほぼ平均画像になる
```

---

## Test 9: Branch dominance sanity

監視:

```text
semantic_bpp / total_bpp
detail_bpp / total_bpp
hyper_bpp / total_bpp
```

異常:

```text
detail branchがtotal bppの70%以上を常に使う
semantic branchがほぼ0情報になる
hyperprior overheadが大きすぎる
```

---

## Test 10: Diffusion hallucination sanity

簡易チェック:

```text
x_aux vs x_hat のDINO similarity
text-like crop visual check
face crop visual check
object count visual check
```

FIDだけが改善してsemantic metricsが悪化する場合、diffusion hallucinationの可能性が高い。
