# Project Milestones

ここでは時期ではなく、達成条件ベースでmilestoneを定義する。

---

## Milestone 0: Environment and baseline audit

成果物:

```text
baseline repo/source inventory
metric pipeline
actual bpp calculation script
Kodak/CLIC/DIV2K loaders
logging template
```

合格条件:

```text
少なくともCompressAI baselineと1つのgenerative baselineを評価できる
```

---

## Milestone 1: Semantic VQ branch

成果物:

```text
SemanticEncoder
DifferentiableVQ
SemanticAuxDecoder
Stage1 checkpoint
semantic-only reconstructions
codebook usage report
```

合格条件:

```text
semantic-only reconstructionが大域構造を保持し、codebook collapseしていない
```

---

## Milestone 2: Semantic entropy coding

成果物:

```text
SemanticEntropyModel
semantic actual bitstream
estimated vs actual bpp report
```

合格条件:

```text
semantic tokenがlossless round-tripし、actual bpp差分が許容内
```

---

## Milestone 3: Detail residual branch

成果物:

```text
DetailResidualEncoder
DetailEntropyModel
JointAuxDecoder
semantic+detail reconstructions
branch bpp decomposition
```

合格条件:

```text
semantic-onlyよりLPIPS/DISTSまたはcritical detailが改善
```

---

## Milestone 4: Diffusion decoder adaptation

成果物:

```text
one-step diffusion decoder adapter
x_hat reconstructions
x_aux vs x_hat comparison
```

合格条件:

```text
perceptual metricsが改善し、faithfulnessが大きく悪化しない
```

---

## Milestone 5: MVP benchmark

成果物:

```text
0.01 / 0.03 / 0.05 bpp results
Kodak / CLIC / DIV2K metrics
baseline comparison
core ablation
visual/failure gallery
```

合格条件:

```text
MVP完了条件を満たす
```

---

## Milestone 6: Full extensions

候補:

```text
variable-rate
adaptive quantization
critical detail metrics
VLM-DPO
human 2AFC
```

合格条件:

```text
投稿候補条件を満たす
```
