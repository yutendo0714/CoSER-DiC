# License, IP, and External Code Policy

## 基本方針

CoSER-DiC本体のcodebaseは新規に構築する。ただし、既存公開実装は以下に利用してよい。

```text
baseline reproduction
pretrained initialization
architecture reference
metric/evaluation reference
entropy coder参考実装
```

---

## 必ず確認すること

各外部repoについて以下を記録する。

```text
license
checkpoint usage terms
commercial/non-commercial restrictions
dataset restrictions
third-party dependencies
citation requirements
```

---

## 禁止/注意

```text
license不明コードをCoSER-DiC本体へ無断コピーしない
外部repoのcheckpointを使った場合は明記する
baseline codeの変更点を記録しないまま結果を出さない
third-party metric実装のversionを記録しないまま比較しない
```

---

## Paperでの表記

外部実装を利用した場合は、以下を明記する。

```text
which codebase
which checkpoint
whether official or reimplementation
whether actual bpp is available
whether metrics are recomputed in our pipeline
```

---

## CoSER-DiC本体の新規性を守る

外部実装の単純改造に見えないよう、以下を独自実装・独自統合として明確にする。

```text
semantic-residual bitstream design
semantic-conditioned residual entropy model
actual bpp decomposition
stage-wise training protocol
ablation/evaluation protocol
```
