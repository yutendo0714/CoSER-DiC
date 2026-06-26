# Dataset Protocol

## Training datasets

推奨順:

```text
OpenImages
COCO
DIV2K train
CLIC train
Flickr2K
```

目的別:

| Dataset | 目的 |
|---|---|
| OpenImages | 多様な自然画像・scene |
| COCO | object layout / semantic consistency |
| DIV2K | high-quality natural image |
| CLIC | compression benchmark整合 |
| Flickr2K | texture / high-resolution補強 |

Full版で追加:

```text
FFHQ / CelebA-HQ: face identity評価・補助訓練
TextCaps / OCR-rich subset: text preservation評価
screen content subset: UI / text robustness
```

---

## Evaluation datasets

必須:

```text
Kodak
CLIC2020 validation/professional
DIV2K validation
```

推奨追加:

```text
Tecnick
COCO val
FFHQ / CelebA-HQ
Text-rich subset
screen content subset
```

---

## Dataset governance

各datasetについて以下を記録する。

```text
source URL
license / terms
version or download date
split definition
preprocessing
excluded files
checksum if feasible
```

---

## Preprocessing rules

Training:

```text
random crop
horizontal flip
RGB [0,1]
no color jitter initially
```

Evaluation:

```text
full-resolution only
no center crop
pad to multiple of 64
remove padding before metrics
```

---

## Dataset-specific warnings

```text
Kodak: 24枚なのでFID/KIDの主張には使わない。
CLIC/DIV2K: FID/KIDに適するが、splitを明記する。
Face/Text subsets: main benchmarkではなくfaithfulness補助として扱う。
```
