# 060 Stage 4 Perceptual Adapter and CoSER Feature Ablation

Date: 2026-06-29 JST

## Decision

Superseded by
`061_stage4_detailcontrast_and_stage5_sparse_control.md` for the current
mainline anchor. This document remains the record for the previous perceptual
adapter and the CoSER feature ablations that justified continuing along the
semantic-latent/detail-context adapter path.

Promote `20260629_stage4_detailfilm_dists_perceptual_ft1200_b8` as the current
best no-extra-bit Stage 4 CoD-Lite adapter anchor.

Promote the matching sparse top-k32 counted-control point as the current best
internal counted-control Stage 5 candidate.

Do not claim external-baseline superiority. These results improve CoSER's own
Stage 4/5 path, but official CoD-Lite perceptual metrics remain far ahead.

## No-Extra-Bit Stage 4 Anchor

The new adapter continues from the DISTS-balanced anchor and raises perceptual
weights while keeping Stage 3 guards:

```text
checkpoint:
  checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailfilm_dists_perceptual_ft1200_b8.pt

full552 eval:
  results/stage4_cod_lite_adapter_eval/20260629_stage4_detailfilm_dists_perceptual_ft1200_b8_full552_eval/summary.json

actual_payload_bpp:
  0.013999

metrics:
  PSNR     21.2237
  MS-SSIM   0.7162
  LPIPS     0.4159
  DISTS     0.2830
  condition_l1 0.4124
```

Delta versus `20260629_stage4_detailfilm_stats_hf_dists_ft1000_b8_full552_eval`:

```text
PSNR     +0.0683
MS-SSIM  +0.0004
LPIPS    -0.0039
DISTS    -0.0046
```

This is a mainline improvement because it changes condition recovery from
decoded CoSER features, not a post-hoc RGB blend.

## Counted-Control Stage 5 Candidate

A new post-affine G32/S8/K256 basis was fitted from the promoted adapter:

```text
basis:
  outputs/stage5_control_basis/20260629_perceptual_detailfilm_postaffine_basis_g32s8_k256_train8192_b12/control_basis.pt

coefficient p99:
  0.710860
```

The best low-control-bpp point is sparse top-k32 Huffman/mu-law:

```text
run:
  20260629_stage5_perceptual_g32basis_full552_affinebasis_ag16_as1_bg32_bs8_k32_b4_topkc256_gr1_br0.25_r0.710860_sc1_huffman_mu_law_mu16_topk_c256_k32_p99_b4_mulaw16

actual_payload_bpp:
  0.015485

control_payload_bpp:
  0.001486

metrics:
  PSNR     21.1592
  MS-SSIM   0.7148
  LPIPS     0.3918
  DISTS     0.2707
  condition_l1 0.4036
```

Delta versus the no-extra promoted adapter:

```text
bpp      +0.001486
PSNR     -0.0645
MS-SSIM  -0.0015
LPIPS    -0.0240
DISTS    -0.0123
```

Delta versus the previous best stats-HF top-k32 counted-control anchor:

```text
bpp      +0.000003
PSNR     +0.0985
MS-SSIM  +0.0020
LPIPS    -0.0001
DISTS    -0.0019
```

Therefore, the new point is the current best internal counted-control anchor:
it slightly improves LPIPS, improves DISTS more clearly, and recovers fidelity
relative to the previous top-k32 point at essentially the same bpp.

## Top-k64 Non-Promotion

Limit64 screening for top-k64:

```text
top-k32:
  bpp 0.015143
  control_bpp 0.001489
  PSNR 21.5510
  MS-SSIM 0.7379
  LPIPS 0.3721
  DISTS 0.2688

top-k64:
  bpp 0.016105
  control_bpp 0.002450
  PSNR 21.5596
  MS-SSIM 0.7380
  LPIPS 0.3715
  DISTS 0.2687
```

Top-k64 costs about +0.00096 bpp but only gives a tiny LPIPS/DISTS improvement
on limit64. Do not promote it as the main point. It can remain a curve point if
we later need denser RD/RP curves.

## CoSER Feature Ablation

Limit64 ablation confirms the adapter is not just a generic CoD-Lite prior:

```text
normal:
  PSNR 21.6103, MS-SSIM 0.7395, LPIPS 0.3945, DISTS 0.2807, condition_l1 0.4118

delta_zero:
  PSNR -0.0235, MS-SSIM -0.0067, LPIPS +0.1602, DISTS +0.0839, condition_l1 +0.1170

semantic_zero:
  PSNR -0.0788, MS-SSIM -0.0041, LPIPS +0.0016, DISTS +0.0007, condition_l1 +0.0007

semantic_shuffle:
  PSNR -0.1039, MS-SSIM -0.0062, LPIPS +0.0055, DISTS +0.0009, condition_l1 +0.0047

detail_zero:
  PSNR +0.0420, MS-SSIM +0.0000, LPIPS +0.0027, DISTS +0.0028, condition_l1 +0.0001

detail_shuffle:
  PSNR +0.0215, MS-SSIM -0.0002, LPIPS +0.0018, DISTS +0.0021, condition_l1 +0.0001
```

Interpretation:

```text
adapter delta:
  essential for perceptual quality and condition recovery.

semantic latent:
  useful for both fidelity and perceptual quality; shuffling is worse than zeroing.

detail context:
  mostly helps LPIPS/DISTS but slightly hurts PSNR, so future detail work should
  make the detail stream more diffusion-friendly rather than merely stronger.
```

## Current Research Consequence

The next mainline target is not more output blending. The highest-value next
step is to improve the information flow before or inside the condition adapter:

```text
1. Add a cleaner diffusion-control detail head from the same transmitted detail
   payload.
2. Train/evaluate it first in no-extra-bit mode.
3. Refit sparse counted-control only after the no-extra adapter improves.
4. Keep all transmitted image-specific controls counted in actual_payload_bpp.
```

This keeps the Core MVP story centered on semantic/detail bitstreams controlling
a pretrained diffusion codec backbone.
