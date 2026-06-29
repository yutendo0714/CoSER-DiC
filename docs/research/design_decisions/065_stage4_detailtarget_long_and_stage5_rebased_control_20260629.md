# 065 Stage 4 Detail-Target Long Run and Stage 5 Rebased Control

Date: 2026-06-29 JST

## Decision

Promote `20260629_stage4_detailtarget_perceptual_long_ft1800_b8` as the
current no-extra-bit Stage 4 CoD-Lite adapter anchor.

Track
`20260629_stage5_detailtarget_perceptual_long_g32basis_postgate_controlaware_perceptual_full552...topk_c256_k32...`
as the current strongest internal Stage 5 perceptual counted-control candidate,
but do not promote it as an external-baseline win.

Promote
`20260629_stage5_detailtarget_perceptual_long_g32basis_postgate_balanced_full552...topk_c256_k32...`
as the current balanced Stage 5 counted-control anchor. It preserves most of
the fixed-control fidelity while giving a small deterministic learned-gate
gain over fixed control scale 1.0.

This update supersedes the earlier detail-contrast anchor in decision 061 for
active continuation. Decision 061 remains useful historical context for the
first sparse counted-control path.

## Why This Is Mainline

The key change is still inside the CoSER-to-CoD-Lite information path:

```text
decoded semantic/detail actual bitstream
  -> semantic latent + detail context
  -> CoSER condition adapter
  -> optional entropy-coded condition-control residual
  -> frozen official CoD-Lite backbone
```

The Stage 4 loss now explicitly asks the decoded detail payload to explain the
CoD-Lite condition residual:

```text
pred_detail_residual = pred_cond - pred_cond_with_zero_detail
target_condition_residual = target_cond - base_cond
```

This is training-only teacher supervision. Inference still uses only decoded
CoSER tensors, fixed CoSER weights, fixed CoD-Lite weights, and any explicitly
counted control stream.

## Current No-Extra-Bit Stage 4 Anchor

Run:

```text
20260629_stage4_detailtarget_perceptual_long_ft1800_b8
```

Checkpoint:

```text
checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailtarget_perceptual_long_ft1800_b8.pt
```

Full552 evaluation:

```text
results/stage4_cod_lite_adapter_eval/20260629_stage4_detailtarget_perceptual_long_ft1800_b8_full552_eval/summary.json
```

Metrics:

| metric | value |
|---|---:|
| actual_payload_bpp | 0.013999 |
| PSNR | 21.3043 |
| MS-SSIM | 0.7166 |
| LPIPS-Alex | 0.4080 |
| DISTS | 0.2780 |
| condition L1 | 0.4104 |

Delta versus the previous no-extra detail-contrast anchor:

| metric | delta |
|---|---:|
| actual_payload_bpp | +0.000000 |
| PSNR | +0.0692 |
| MS-SSIM | -0.0000 |
| LPIPS-Alex | -0.0059 |
| DISTS | -0.0040 |
| condition L1 | -0.0004 |

Split deltas versus the previous no-extra anchor:

| split | PSNR | MS-SSIM | LPIPS | DISTS | condition L1 |
|---|---:|---:|---:|---:|---:|
| Kodak24 | +0.0644 | -0.0001 | -0.0047 | -0.0030 | +0.0013 |
| CLIC2020 test428 | +0.0712 | -0.0000 | -0.0055 | -0.0041 | -0.0004 |
| DIV2K val100 | +0.0620 | -0.0000 | -0.0077 | -0.0041 | -0.0006 |

Interpretation:

```text
This is a real no-extra-bit Stage 4 improvement.
It is still not close to official CoD-Lite perceptual quality.
```

## Rebased Sparse Stage 5 Control

A new post-affine G32/S8/K256 residual basis was fitted from the updated Stage
4 adapter:

```text
outputs/stage5_control_basis/20260629_detailtarget_perceptual_long_postaffine_basis_g32s8_k256_train8192_b12/control_basis.pt
```

Coefficient range:

```text
p99 = 0.696237
```

The sparse top-k32 Huffman/mu-law stream remains tiny and counted:

```text
control_payload_bpp ~= 0.001485
actual_payload_bpp ~= 0.015484
```

### Fixed Control Scale Points

| point | actual bpp | PSNR | MS-SSIM | LPIPS | DISTS | condition L1 |
|---|---:|---:|---:|---:|---:|---:|
| scale 1.00 | 0.015484 | 21.2344 | 0.7147 | 0.3844 | 0.2662 | 0.4016 |
| scale 1.25 | 0.015485 | 21.2049 | 0.7139 | 0.3802 | 0.2638 | 0.4025 |

Fixed scale remains an operating-point diagnostic, not the final method.

### Control-Aware Post-Control Gate

Run:

```text
20260629_stage5_post_control_condition_gate_detailtarget_long_controlaware_perceptual_ft1200_b4
```

Full552 evaluation:

```text
results/stage4_cod_lite_adapter_eval/20260629_stage5_detailtarget_perceptual_long_g32basis_postgate_controlaware_perceptual_full552_affinebasis_ag16_as1_bg32_bs8_k32_b4_topkc256_gr1_br0.25_r0.696237_sc1_huffman_mu_law_mu16_topk_c256_k32_p99_b4_mulaw16/summary.json
```

Metrics:

| metric | value |
|---|---:|
| actual_payload_bpp | 0.015484 |
| control_payload_bpp | 0.001485 |
| PSNR | 21.1337 |
| MS-SSIM | 0.7101 |
| LPIPS-Alex | 0.3696 |
| DISTS | 0.2572 |
| condition L1 | 0.4043 |
| post-control gate mean | 1.1055 |

Delta versus the previous detail-contrast Stage 5 postgate:

| metric | delta |
|---|---:|
| actual_payload_bpp | -0.000002 |
| PSNR | +0.0150 |
| MS-SSIM | -0.0025 |
| LPIPS-Alex | -0.0083 |
| DISTS | -0.0063 |
| condition L1 | +0.0014 |

Split deltas versus the previous detail-contrast Stage 5 postgate:

| split | PSNR | MS-SSIM | LPIPS | DISTS | condition L1 |
|---|---:|---:|---:|---:|---:|
| Kodak24 | +0.0159 | -0.0033 | -0.0078 | -0.0078 | +0.0032 |
| CLIC2020 test428 | +0.0161 | -0.0024 | -0.0076 | -0.0064 | +0.0014 |
| DIV2K val100 | +0.0104 | -0.0028 | -0.0115 | -0.0060 | +0.0011 |

Interpretation:

```text
This is the strongest internal perceptual Stage 5 candidate so far.
It improves LPIPS/DISTS on every strict full552 split.
It still loses too much MS-SSIM and remains far from official CoD-Lite.
```

### Balanced Post-Control Gate

Run:

```text
20260629_stage5_post_control_condition_gate_detailtarget_long_balanced_ft1000_b4
```

Full552 evaluation:

```text
results/stage4_cod_lite_adapter_eval/20260629_stage5_detailtarget_perceptual_long_g32basis_postgate_balanced_full552_affinebasis_ag16_as1_bg32_bs8_k32_b4_topkc256_gr1_br0.25_r0.696237_sc1_huffman_mu_law_mu16_topk_c256_k32_p99_b4_mulaw16/summary.json
```

Metrics:

| metric | value |
|---|---:|
| actual_payload_bpp | 0.015484 |
| control_payload_bpp | 0.001485 |
| PSNR | 21.2378 |
| MS-SSIM | 0.7146 |
| LPIPS-Alex | 0.3825 |
| DISTS | 0.2660 |
| condition L1 | 0.4013 |
| post-control gate mean | 1.0046 |

Delta versus fixed control scale 1.0:

| metric | delta |
|---|---:|
| actual_payload_bpp | +0.000000 |
| PSNR | +0.0034 |
| MS-SSIM | -0.0002 |
| LPIPS-Alex | -0.0020 |
| DISTS | -0.0002 |
| condition L1 | -0.0003 |

Split deltas versus fixed control scale 1.0:

| split | PSNR | MS-SSIM | LPIPS | DISTS | condition L1 |
|---|---:|---:|---:|---:|---:|
| Kodak24 | +0.0148 | +0.0003 | -0.0011 | +0.0010 | -0.0005 |
| CLIC2020 test428 | +0.0017 | -0.0002 | -0.0021 | -0.0004 | -0.0003 |
| DIV2K val100 | +0.0079 | +0.0001 | -0.0015 | +0.0003 | -0.0003 |

Interpretation:

```text
This is the better current Stage 5 anchor for balanced continuation.
The gain over fixed scale is modest but real, counted-bpp-neutral, and learned
from decoded CoSER/control tensors rather than posthoc RGB edits.
The stronger perceptual gate remains useful as an upper-pressure direction,
but it is too destructive to be the balanced anchor.
```

## Balanced Stage 5 Feature-Use Ablations

The current balanced anchor was audited with full552 ablations at the same
strict 512 protocol.

Semantic latent shuffle:

```text
results/stage4_cod_lite_adapter_eval/20260629_stage5_detailtarget_long_balanced_full552_semshuffle_ablation/summary.json
```

Detail context zero:

```text
results/stage4_cod_lite_adapter_eval/20260629_stage5_detailtarget_long_balanced_full552_detailzero_ablation/summary.json
```

Delta versus the balanced Stage 5 anchor:

| ablation | PSNR | MS-SSIM | LPIPS | DISTS | condition L1 |
|---|---:|---:|---:|---:|---:|
| semantic latent shuffle | -0.1395 | -0.0076 | +0.0083 | +0.0035 | +0.0053 |
| detail context zero | +0.0400 | -0.0001 | +0.0111 | +0.0054 | +0.0062 |

Interpretation:

```text
The balanced Stage 5 path clearly uses the decoded semantic latent.
The decoded detail context is still not a fidelity booster; zeroing it slightly
raises PSNR but hurts LPIPS, DISTS, and condition L1. This supports the next
mainline move: make the detail payload more diffusion-control-friendly instead
of treating it as only an RGB residual.
```

## Initial Actual-Bpp Control Curve

The same balanced post-control gate checkpoint was evaluated with top-k16,
top-k24, top-k32, and top-k48 sparse control streams. This is a provisional
curve because the gate was trained on the top-k32 control distribution, but it
uses actual entropy-coded payload bpp for every point.

| point | actual bpp | control bpp | PSNR | MS-SSIM | LPIPS | DISTS | condition L1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-extra Stage 4 | 0.013999 | 0.000000 | 21.3043 | 0.7166 | 0.4080 | 0.2780 | 0.4104 |
| top-k16 | 0.014977 | 0.000978 | 21.2341 | 0.7144 | 0.3833 | 0.2662 | 0.4020 |
| top-k24 | 0.015233 | 0.001234 | 21.2351 | 0.7144 | 0.3828 | 0.2660 | 0.4016 |
| top-k32 | 0.015484 | 0.001485 | 21.2378 | 0.7146 | 0.3825 | 0.2660 | 0.4013 |
| top-k48 | 0.015984 | 0.001985 | 21.2392 | 0.7145 | 0.3820 | 0.2658 | 0.4009 |

Artifacts:

```text
results/stage4_cod_lite_adapter_eval/20260629_stage5_detailtarget_long_balanced_full552_topk16_curve/summary.json
results/stage4_cod_lite_adapter_eval/20260629_stage5_detailtarget_long_balanced_full552_topk24_curve/summary.json
results/stage4_cod_lite_adapter_eval/20260629_stage5_detailtarget_perceptual_long_g32basis_postgate_balanced_full552_affinebasis_ag16_as1_bg32_bs8_k32_b4_topkc256_gr1_br0.25_r0.696237_sc1_huffman_mu_law_mu16_topk_c256_k32_p99_b4_mulaw16/summary.json
results/stage4_cod_lite_adapter_eval/20260629_stage5_detailtarget_long_balanced_full552_topk48_curve/summary.json
```

Top-k16-specific gate:

```text
checkpoint:
  checkpoints/stage4_cod_lite_condition_gate/20260629_stage5_post_control_condition_gate_detailtarget_long_topk16_balanced_ft800_b4.pt

evaluation:
  results/stage4_cod_lite_adapter_eval/20260629_stage5_detailtarget_long_topk16_specific_gate_full552/summary.json
```

Delta versus using the top-k32-trained balanced gate at the same top-k16
payload:

| metric | delta |
|---|---:|
| actual_payload_bpp | +0.000000 |
| PSNR | -0.0127 |
| MS-SSIM | -0.0005 |
| LPIPS-Alex | -0.0014 |
| DISTS | -0.0012 |
| condition L1 | +0.0003 |

Interpretation:

```text
The first approximately 0.001 bpp of counted control gives most of the current
LPIPS/DISTS gain. Higher top-k mostly improves condition L1 slightly and gives
only small image-metric gains. The next high-value move is therefore not just
adding more sparse coefficients; it is improving the control representation,
the detail-to-control head, or retraining gates/adapters for each rate point.

The top-k16-specific gate result supports rate-specific gate training: it gains
LPIPS/DISTS at equal actual bpp, with a small fidelity cost.
```

## GenCodec-Style Patch-FID / KID

Saved reconstructions:

```text
results/stage4_cod_lite_adapter_eval/20260629_stage5_detailtarget_perceptual_long_g32basis_postgate_controlaware_perceptual_full552_save_affinebasis_ag16_as1_bg32_bs8_k32_b4_topkc256_gr1_br0.25_r0.696237_sc1_huffman_mu_law_mu16_topk_c256_k32_p99_b4_mulaw16/reconstructions/
```

Patch protocol:

```text
Kodak:
  patch_size 64, fid_patch_num 2

CLIC2020 test / DIV2K val:
  patch_size 256, fid_patch_num 2
```

Patch extraction is used only for FID/KID. PSNR, MS-SSIM, LPIPS, and DISTS are
still full-image metrics from the Stage 4/5 evaluator.

| split | Stage3 FID | Stage5 FID | FID delta | Stage3 KID | Stage5 KID | KID delta |
|---|---:|---:|---:|---:|---:|---:|
| Kodak24 | 216.3567 | 77.1709 | -139.1858 | 0.150822 | 0.019143 | -0.131679 |
| CLIC2020 test428 | 152.0035 | 56.3215 | -95.6820 | 0.079694 | 0.006421 | -0.073273 |
| DIV2K val100 | 234.6295 | 151.8839 | -82.7455 | 0.092627 | 0.012842 | -0.079785 |

Balanced post-control gate patch-FID/KID:

| split | FID | KID |
|---|---:|---:|
| Kodak24 | 81.1976 | 0.022353 |
| CLIC2020 test428 | 57.6185 | 0.006986 |
| DIV2K val100 | 153.6698 | 0.013911 |

Interpretation:

```text
The perceptual gate is better for FID/KID.
The balanced gate is the safer anchor because it keeps much more fidelity and
still gives large distribution gains versus Stage 3.
```

Artifacts:

```text
results/distribution_metrics/20260629_stage5_detailtarget_long_postgate_kodak_gencodec_patch64_n2_fid_kid.json
results/distribution_metrics/20260629_stage5_detailtarget_long_postgate_clic_gencodec_patch256_n2_fid_kid.json
results/distribution_metrics/20260629_stage5_detailtarget_long_postgate_div2k_gencodec_patch256_n2_fid_kid.json
results/distribution_metrics/20260629_stage5_detailtarget_long_balanced_kodak_gencodec_patch64_n2_fid_kid.json
results/distribution_metrics/20260629_stage5_detailtarget_long_balanced_clic_gencodec_patch256_n2_fid_kid.json
results/distribution_metrics/20260629_stage5_detailtarget_long_balanced_div2k_gencodec_patch256_n2_fid_kid.json
results/distribution_metrics/20260629_stage3_from_stage5_save_kodak_gencodec_patch64_n2_fid_kid.json
results/distribution_metrics/20260629_stage3_from_stage5_save_clic_gencodec_patch256_n2_fid_kid.json
results/distribution_metrics/20260629_stage3_from_stage5_save_div2k_gencodec_patch256_n2_fid_kid.json
```

Important audit note:

```text
Use reconstructions/manifest.jsonl for saved reconstruction FID/KID.
Do not use per_image_metrics.jsonl as the GenCodec patch manifest, because it
does not contain the saved reference/stage3/stage4 filenames.
```

## External Baseline Gap

Official CoD-Lite Kodak512 at nominal 0.015625 bpp reports:

```text
PSNR 20.7667
MS-SSIM 0.7090
LPIPS 0.2259
DISTS 0.1402
FID 38.4785
```

The current CoSER Stage 5 point is much better than Stage 3 on perceptual and
distribution metrics, but it is still not externally competitive on LPIPS,
DISTS, or FID. Do not claim curve superiority.

The honest current position is:

```text
CoSER has a stable actual-bitstream semantic/detail substrate.
Stage 4 now improves no-extra-bit condition/image quality.
Stage 5 counted control gives large internal perceptual/distribution gains.
The external CoD-Lite/RDVQ/GLC/StableCodec curve gap remains large.
```

## Next Mainline Actions

1. Add control-aware ablations: no semantic latent, no detail context, shuffled
   detail, zero control, shuffled control, and gate frozen to 1.0.
2. Build curve points from the same updated Stage 4 anchor: no-extra, top-k16,
   top-k24, top-k32, and top-k48 counted control.
3. Improve the diffusion-friendly detail head from the same transmitted detail
   payload before increasing counted side information.
4. Train a stronger balanced gate or condition adapter that reduces the gap
   between the balanced and perceptual gate without sacrificing MS-SSIM.
5. Run matched official CoD-Lite full552 baselines using the same strict 512
   protocol before any paper-style claim.

## Bottom Line

This is meaningful mainline progress:

```text
same no-extra payload improved Stage 4
tiny counted control improved Stage 5 perceptual/distribution metrics
GenCodec-style patch-FID/KID improved strongly over Stage 3 on all splits
```

But this is not the large external-baseline curve win yet. The next research
should continue along condition recovery, control-aware gating, diffusion-
friendly detail representation, and actual-payload curve construction.
