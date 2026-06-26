# Evaluation Protocol

評価条件が曖昧だと、生成圧縮の比較は信用されない。この文書のルールを固定する。

---

## General rules

```text
評価はcompress/decompress APIを通す。
forward-onlyのestimated bpp評価はdebug用途のみ。
評価時はfull-resolution。
paddingはpad_to_multiple=64。
metric計算前に元サイズへcrop。
color spaceとrangeを統一する。
```

---

## Kodak

```yaml
dataset: Kodak
num_images: 24
evaluation: full_resolution
crop: false
padding: pad_to_multiple_64
remove_padding_before_metric: true
metrics:
  - actual_bpp
  - PSNR_RGB
  - PSNR_Y
  - MS_SSIM_RGB
  - LPIPS_Alex
  - DISTS
```

注意:

```text
KodakでFIDを主張しない。
24枚しかないためFIDの信頼性が低い。
```

---

## CLIC / DIV2K

```yaml
datasets:
  - CLIC2020 validation/professional
  - DIV2K validation
metrics:
  - actual_bpp
  - PSNR_RGB
  - MS_SSIM
  - LPIPS_Alex
  - DISTS
  - FID
  - KID
```

FID / KIDはCLICやDIV2Kのような枚数が多いdatasetで主に見る。

---

## Tecnick / high-resolution

目的:

```text
高解像度でruntimeとmemory、padding処理、arbitrary resolution対応を確認する。
```

metrics:

```text
actual_bpp
PSNR/MS-SSIM
LPIPS/DISTS
runtime/memory
```

---

## Semantic-critical evaluation

Full版では以下を追加する。

```text
OCR subset
face subset
small object subset
text-rich images
screen-content images
```

metrics:

```text
OCR recognition accuracy
OCR edit distance
face identity cosine similarity
DINO feature similarity
CLIP image similarity
segmentation consistency
edge F-score
object count consistency
critical-region LPIPS / DISTS
```

---

## Metrics implementation rules

推奨:

```text
PSNR_RGB: RGB [0,1] or [0,255] conversionを明記
PSNR_Y: RGB->YCbCr変換の実装を固定
MS-SSIM: same implementation across all methods
LPIPS: Alex backboneをdefault。VGGを使うなら別表記。
DISTS: official implementation versionを記録
FID/KID: torch-fidelity等、実装versionを記録
```

---

## Runtime evaluation

報告項目:

```text
encode time
decode time
arithmetic coding time
diffusion steps
GPU model
CUDA version
batch size
image resolution
peak GPU memory
model parameters
```

one-step diffusionを主張する場合、arithmetic coding時間込みで測る。

---

## Human evaluation

2AFC形式を推奨する。

質問は必ず2種類に分ける。

```text
Q1: Which image is more visually pleasing?
Q2: Which image is more faithful to the original?
```

理由:

```text
generative codecは見た目では勝っても、faithfulnessでは負ける可能性がある。
この2つを混ぜると研究の主張が曖昧になる。
```
