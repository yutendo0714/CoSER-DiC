# Figure and Table Plan

## Figure 1: Overall architecture

Show:

```text
input -> semantic VQ -> semantic bitstream
input + x_sem -> detail residual -> detail bitstream
semantic/detail -> x_aux -> one-step diffusion -> x_hat
```

Highlight role separation.

---

## Figure 2: Information allocation

Visualize:

```text
semantic token map
detail residual map
x_sem
x_aux
x_hat
```

---

## Figure 3: Rate-Distortion-Perception curves

Plot:

```text
bpp vs LPIPS
bpp vs DISTS
bpp vs PSNR
bpp vs FID/KID where dataset supports
```

---

## Figure 4: Visual comparison

Show original and reconstructions at matched bpp.

---

## Figure 5: Failure and correction examples

Show how detail residual improves:

```text
text
face
small object
object boundary
```

---

## Table 1: Main quantitative comparison

Kodak/CLIC/DIV2K.

---

## Table 2: Ablation

Core variants.

---

## Table 3: Semantic faithfulness

DINO/CLIP/OCR/face/segmentation.

---

## Table 4: Runtime and model size

Encode/decode/arithmetic coding/memory/params.

---

## Appendix figures

```text
additional visual comparisons
more failure cases
rate allocation heatmaps
codebook usage plots
estimated vs actual bpp calibration
```
