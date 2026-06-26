# Baseline Reproduction Protocol

## 方針

比較対象は多いが、最初から全てを再現しない。MVPでは直接競合に絞る。

---

## Priority A: 必須baseline

```text
GLC
RDVQ
ResULIC
RDEIC
StableCodec or CoD-Lite
BPG
VVC/VTM
CompressAI hyperprior or Cheng2020
```

これらはMVP時点で最低限比較したい。

---

## Priority B: 可能なら比較

```text
Control-GIC
DLF
OneDC
AEIC
TCM
MLIC++
```

---

## Priority C: 参考・論文中の議論

```text
HiFiC
MS-ILLM
EGIC
PerCo
TACO
DiffC
CADC
VLIC
MRIDC
```

CADCとVLICは非常に重要だが、実装公開状況や評価条件によっては論文値・設計比較に留める。

---

## 比較条件

```text
same evaluation images
same image size
same color space
same metric implementation
same bpp definition where possible
actual bpp preferred
official checkpoint preferred
```

論文値引用は補助扱い。main tableには原則として同一pipelineで再評価した値を載せる。

---

## 公式checkpoint vs retraining

優先順位:

```text
1. 公式checkpoint + 公式compress/decompressでbitstream生成
2. 公式checkpoint + 自前metric pipelineで評価
3. 公式実装を同一datasetで再学習
4. 論文値引用
```

4はmain comparisonではなく参考にする。

---

## actual bppが取れないbaseline

actual bppが取れないbaselineは、main tableでの主張に使わない。

どうしても使う場合:

```text
estimated bppであることを明記
別表または参考比較に分ける
同一bpp主張を避ける
```

---

## bpp matching

各手法について複数rate pointを評価し、RD/RDP curveを作る。

近傍比較:

```text
target bpp ±5%以内を優先
±10%を超える場合は同一rate比較として扱わない
```

---

## Baseline reproduction log

各baselineについて以下を記録する。

```yaml
method:
repo_url:
paper_url:
commit_hash:
checkpoint:
license_checked:
actual_bpp_available:
metric_pipeline:
preprocessing:
runtime_environment:
notes:
```
