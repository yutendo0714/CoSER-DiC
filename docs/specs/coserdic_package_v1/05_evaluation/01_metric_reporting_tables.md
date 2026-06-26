# Metric Reporting Tables

論文用に最低限、以下の表を作る。

---

## Table 1: Kodak RD/RDP performance

```text
Method | actual bpp | PSNR-RGB | PSNR-Y | MS-SSIM | LPIPS-Alex | DISTS
```

KodakではFIDを主指標にしない。

---

## Table 2: CLIC/DIV2K perceptual performance

```text
Method | actual bpp | LPIPS | DISTS | FID | KID | PSNR | MS-SSIM
```

FID/KIDは十分な枚数のdatasetで計算する。

---

## Table 3: Ultra-low bitrate comparison

```text
Method | 0.003 bpp | 0.005 bpp | 0.01 bpp | 0.03 bpp | 0.05 bpp
```

各cellにはDISTSまたはLPIPSを入れる。別表でFID/KIDも示す。

---

## Table 4: Semantic faithfulness

```text
Method | actual bpp | DINO sim | CLIP sim | seg consistency | OCR acc | face ID sim | edge F-score
```

---

## Table 5: Efficiency

```text
Method | actual bpp | encode time | decode time | arithmetic coding time | params | memory | diffusion steps
```

---

## Table 6: Ablation

```text
Variant | actual bpp | LPIPS | DISTS | FID/KID | OCR | face ID | decode time
```

必須variant:

```text
semantic only
semantic + detail
semantic + diffusion
semantic + detail + diffusion
hard VQ vs soft RDVQ-style VQ
no semantic-conditioned residual prior
no auxiliary decoder
```
