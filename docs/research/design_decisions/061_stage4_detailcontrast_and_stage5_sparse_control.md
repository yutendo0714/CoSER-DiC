# 061 Stage 4 Detail-Contrast Adapter and Stage 5 Sparse Control

Date: 2026-06-29 JST

## Decision

Promote `20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8` as the current
fidelity-tilted no-extra-bit Stage 4 CoD-Lite adapter anchor.

Promote `20260629_stage4_condition_strength_gate_balanced_spatial_ft800_b2ga2`
as the current no-extra-bit balanced condition-strength gate operating point.
This gate is deterministic from decoded CoSER tensors and fixed model weights;
it transmits no image-specific side information and therefore leaves
`actual_payload_bpp` unchanged.

Promote the matching sparse top-k32 counted-control point as the current best
internal Stage 5 counted-control anchor. Also keep its `control_scale=1.25`
variant as the current perceptual-oriented operating point.

Track the new post-control deterministic condition gate as the current most
perceptual Stage 5 control operating point. It improves LPIPS/DISTS beyond
`control_scale=1.25` at effectively the same counted payload, and improves
Kodak/CLIC patch-FID/KID, but it loses additional fidelity and slightly worsens
DIV2K patch-FID/KID. Therefore it is a mainline research signal, not the
balanced anchor.

Do not claim external-baseline superiority. Official CoD-Lite remains far ahead
on Kodak perceptual/FID numbers at comparable nominal rate. These results are
mainline internal progress: they improve condition recovery and diffusion
control from CoSER semantic/detail payloads.

## No-Extra-Bit Stage 4 Adapter Anchor

The adapter continues from:

```text
checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailfilm_dists_perceptual_ft1200_b8.pt
```

and adds detail-contrast / high-frequency residual losses while preserving
Stage 3 guards.

```text
checkpoint:
  checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8.pt

full552 eval:
  results/stage4_cod_lite_adapter_eval/20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8_full552_eval/summary.json

actual_payload_bpp:
  0.013999

metrics:
  PSNR     21.2351
  MS-SSIM   0.7166
  LPIPS     0.4139
  DISTS     0.2821
  condition_l1 0.4108
```

Delta versus the previous no-extra perceptual anchor:

```text
PSNR     +0.0114
MS-SSIM  +0.0004
LPIPS    -0.0020
DISTS    -0.0009
condition_l1 -0.0017
```

This is a small but clean all-metric full552 improvement at unchanged
`actual_payload_bpp`.

## No-Extra-Bit Balanced Condition Gate

The first learned condition-strength gate was intentionally allowed to amplify
the adapter residual:

```text
checkpoint:
  checkpoints/stage4_cod_lite_condition_gate/20260629_stage4_condition_strength_gate_amp_spatial_ft800_b1ga4.pt

full552 eval:
  results/stage4_cod_lite_adapter_eval/20260629_stage4_condition_strength_gate_amp_spatial_ft800_b1ga4_full552_eval/summary.json

actual_payload_bpp:
  0.013999

metrics:
  PSNR     21.1326
  MS-SSIM   0.7123
  LPIPS     0.3963
  DISTS     0.2712
  condition_l1 0.4131
  condition_gate_mean 1.1374
```

This confirmed that decoder-side condition-strength control is meaningful, but
it is too perceptual-heavy for the balanced no-extra-bit anchor because it
costs about 0.10 dB PSNR and 0.0043 MS-SSIM versus the detail-contrast adapter.

The balanced gate was then retrained with stronger Stage 3/fidelity guards and
a gate mean target near 1.03:

```text
checkpoint:
  checkpoints/stage4_cod_lite_condition_gate/20260629_stage4_condition_strength_gate_balanced_spatial_ft800_b2ga2.pt

full552 eval:
  results/stage4_cod_lite_adapter_eval/20260629_stage4_condition_strength_gate_balanced_spatial_ft800_b2ga2_full552_eval/summary.json

actual_payload_bpp:
  0.013999

metrics:
  PSNR     21.2121
  MS-SSIM   0.7154
  LPIPS     0.4068
  DISTS     0.2786
  condition_l1 0.4106
  condition_gate_mean 1.0488
```

Delta versus the no-extra adapter anchor:

```text
bpp      +0.000000
PSNR     -0.0229
MS-SSIM  -0.0012
LPIPS    -0.0071
DISTS    -0.0034
condition_l1 -0.0002
```

Split deltas versus the no-extra adapter anchor:

```text
Kodak24:
  PSNR -0.0262 / MS-SSIM -0.0018 / LPIPS -0.0101 / DISTS -0.0035

CLIC2020 test428:
  PSNR -0.0230 / MS-SSIM -0.0011 / LPIPS -0.0069 / DISTS -0.0034

DIV2K val100:
  PSNR -0.0221 / MS-SSIM -0.0012 / LPIPS -0.0073 / DISTS -0.0036
```

Interpretation:

```text
The balanced gate is not a posthoc RGB blend.
It is a deterministic condition-space gate over the CoSER-to-CoD-Lite adapter
residual.

It gives a clean no-extra-bit perceptual/distribution operating point, while
the ungated detail-contrast adapter remains the fidelity-tilted no-extra-bit
anchor.
```

Combining the stronger amplification gate with the Stage 5 counted-control
stream was screened on limit64 and was not promoted:

```text
gate + counted control:
  bpp 0.015141
  PSNR 21.5609
  MS-SSIM 0.7378
  LPIPS 0.3710
  DISTS 0.2680
  condition_l1 0.4016

counted control only:
  bpp 0.015147
  PSNR 21.5570
  MS-SSIM 0.7380
  LPIPS 0.3711
  DISTS 0.2675
  condition_l1 0.4012
```

The deltas are too small and mixed; do not run full552 for this combination
unless a new post-control gate design is implemented.

## Counted-Control Anchor

A new post-affine G32/S8/K256 residual basis was fitted from the detail-contrast
adapter on the non-eval 8192-image train cache:

```text
basis:
  outputs/stage5_control_basis/20260629_detailcontrast_hf_postaffine_basis_g32s8_k256_train8192_b12/control_basis.pt

coefficient p99:
  0.702655
```

Sparse top-k32 Huffman/mu-law counted control:

```text
run:
  20260629_stage5_detailcontrast_hf_g32basis_full552_affinebasis_ag16_as1_bg32_bs8_k32_b4_topkc256_gr1_br0.25_r0.702655_sc1_huffman_mu_law_mu16_topk_c256_k32_p99_b4_mulaw16

actual_payload_bpp:
  0.015486

control_payload_bpp:
  about 0.00149

metrics:
  PSNR     21.1617
  MS-SSIM   0.7149
  LPIPS     0.3891
  DISTS     0.2691
  condition_l1 0.4018
```

Delta versus the previous counted-control anchor:

```text
bpp      +0.000001
PSNR     +0.0025
MS-SSIM  +0.0001
LPIPS    -0.0028
DISTS    -0.0015
condition_l1 -0.0018
```

Delta versus the new no-extra anchor:

```text
bpp      +0.001487
PSNR     -0.0734
MS-SSIM  -0.0017
LPIPS    -0.0248
DISTS    -0.0129
condition_l1 -0.0090
```

This is the current best balanced internal Stage 5 counted-control point.

## Perceptual-Oriented Control Scale

The same counted payload with `control_scale=1.25` gives a stronger perceptual
operating point at essentially identical bpp:

```text
run:
  20260629_stage5_detailcontrast_hf_g32basis_full552_affinebasis_ag16_as1_bg32_bs8_k32_b4_topkc256_gr1_br0.25_r0.702655_sc125_huffman_mu_law_mu16_topk_c256_k32_p99_b4_mulaw16

actual_payload_bpp:
  0.015488

metrics:
  PSNR     21.1319
  MS-SSIM   0.7140
  LPIPS     0.3844
  DISTS     0.2663
  condition_l1 0.4027
```

Delta versus `control_scale=1.0`:

```text
bpp      +0.000002
PSNR     -0.0298
MS-SSIM  -0.0008
LPIPS    -0.0047
DISTS    -0.0028
condition_l1 +0.0009
```

Interpretation:

```text
scale=1.0:
  better condition imitation and fidelity.

scale=1.25:
  better LPIPS/DISTS and distribution metrics, with additional fidelity loss.
```

This is a strong reason to replace fixed control scale with a deterministic
content gate learned from decoded CoSER features. The fixed scale itself should
remain an operating-point diagnostic, not the final method.

## Post-Control Condition Gate Probe

The first counted-control gate integration that looks mainline-useful applies
the learned deterministic condition-strength gate after the sparse counted
control correction:

```text
pred_cond = base_cond + gate(decoded CoSER features) * (controlled_cond - base_cond)
```

This is still decoder-side deterministic and transmits no extra gate bits. The
only additional image-specific payload remains the counted sparse control
stream already included in `actual_payload_bpp`.

```text
run:
  20260629_stage5_detailcontrast_hf_g32basis_postgate_balanced_full552_affinebasis_ag16_as1_bg32_bs8_k32_b4_topkc256_gr1_br0.25_r0.702655_sc1_huffman_mu_law_mu16_topk_c256_k32_p99_b4_mulaw16

actual_payload_bpp:
  0.015486

control_payload_bpp:
  about 0.001487

metrics:
  PSNR     21.1249
  MS-SSIM   0.7129
  LPIPS     0.3813
  DISTS     0.2650
  condition_l1 0.4028
  post_control_condition_gate_mean 1.0510
```

Delta versus the `control_scale=1.25` perceptual point:

```text
bpp      -0.000002
PSNR     -0.0070
MS-SSIM  -0.0011
LPIPS    -0.0031
DISTS    -0.0013
condition_l1 +0.0001
```

Split deltas versus `control_scale=1.25`:

```text
Kodak24:
  bpp +0.000015 / PSNR -0.0019 / MS-SSIM -0.0018 / LPIPS -0.0031 / DISTS -0.0004

CLIC2020 test428:
  bpp -0.000003 / PSNR -0.0074 / MS-SSIM -0.0011 / LPIPS -0.0031 / DISTS -0.0014

DIV2K val100:
  bpp -0.000000 / PSNR -0.0066 / MS-SSIM -0.0011 / LPIPS -0.0032 / DISTS -0.0012
```

Interpretation:

```text
The post-control gate is better than the fixed `control_scale=1.25` point for
LPIPS and DISTS on every split. It also improves Kodak/CLIC patch-FID/KID.
However, it sacrifices fidelity and does not improve DIV2K patch-FID/KID.

Keep it as the current best perceptual-control probe and use it to design a
true control-aware gate with stronger fidelity/distribution guards. Do not
promote it as the balanced Stage 5 anchor.
```

## Top-k64 Non-Promotion

Limit64 screening:

```text
top-k32:
  bpp 0.015147
  PSNR 21.5570
  MS-SSIM 0.7380
  LPIPS 0.3711
  DISTS 0.2675
  condition_l1 0.4012

top-k64:
  bpp 0.016127
  PSNR 21.5632
  MS-SSIM 0.7381
  LPIPS 0.3705
  DISTS 0.2673
  condition_l1 0.4005
```

Top-k64 costs about +0.00098 bpp for very small LPIPS/DISTS gains. Keep it as
a possible curve point, but do not promote it as the main Stage 5 candidate.

## Patch-FID / KID

Saved full552 reconstructions:

```text
results/stage4_cod_lite_adapter_eval/20260629_stage5_detailcontrast_hf_g32basis_full552_sc125_save_eval/
results/stage4_cod_lite_adapter_eval/20260629_stage5_detailcontrast_hf_g32basis_postgate_balanced_full552_save_affinebasis_ag16_as1_bg32_bs8_k32_b4_topkc256_gr1_br0.25_r0.702655_sc1_huffman_mu_law_mu16_topk_c256_k32_p99_b4_mulaw16/
```

Patch protocol:

```text
backend:
  torch-fidelity

Kodak:
  patch_size 64, fid_patch_num 2

CLIC2020 test / DIV2K val:
  patch_size 256, fid_patch_num 2
```

Patch extraction is used only for FID/KID. Full-image PSNR, MS-SSIM, LPIPS, and
DISTS remain from the Stage 4/5 full552 evaluator.

| dataset | stage3 FID | no-extra base FID | balanced gate FID | stage5 sc1.25 FID | post-control gate FID | stage3 KID | no-extra base KID | balanced gate KID | stage5 sc1.25 KID | post-control gate KID |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Kodak24 | 216.3567 | 90.8343 | 87.9047 | 79.2004 | 78.6804 | 0.150822 | 0.029419 | 0.027304 | 0.020942 | 0.020334 |
| CLIC2020 test428 | 152.0035 | 61.3787 | 60.6337 | 57.4915 | 57.4315 | 0.079694 | 0.008463 | 0.008175 | 0.007016 | 0.006960 |
| DIV2K val100 | 234.6295 | 160.6237 | 159.1860 | 152.6266 | 152.9545 | 0.092627 | 0.016803 | 0.016311 | 0.013602 | 0.013650 |

Artifacts:

```text
results/analysis/image_distribution_metrics/
20260629_stage5_detailcontrast_sc125_stage4_kodak_gencodec_patch64_n2_fid_kid.json
20260629_stage5_detailcontrast_sc125_stage4_clic2020_test_gencodec_patch256_n2_fid_kid.json
20260629_stage5_detailcontrast_sc125_stage4_div2k_val_gencodec_patch256_n2_fid_kid.json
20260629_stage4_detailcontrast_base_stage4_kodak_gencodec_patch64_n2_fid_kid.json
20260629_stage4_detailcontrast_base_stage4_clic2020_test_gencodec_patch256_n2_fid_kid.json
20260629_stage4_detailcontrast_base_stage4_div2k_val_gencodec_patch256_n2_fid_kid.json
20260629_stage4_balanced_gate_stage4_kodak_gencodec_patch64_n2_fid_kid.json
20260629_stage4_balanced_gate_stage4_clic2020_test_gencodec_patch256_n2_fid_kid.json
20260629_stage4_balanced_gate_stage4_div2k_val_gencodec_patch256_n2_fid_kid.json
20260629_stage5_postgate_balanced_sc1_kodak_gencodec_patch64_n2_fid_kid.json
20260629_stage5_postgate_balanced_sc1_clic2020_test_gencodec_patch256_n2_fid_kid.json
20260629_stage5_postgate_balanced_sc1_div2k_val_gencodec_patch256_n2_fid_kid.json
20260629_stage5_detailcontrast_sc125_stage3_kodak_gencodec_patch64_n2_fid_kid.json
20260629_stage5_detailcontrast_sc125_stage3_clic2020_test_gencodec_patch256_n2_fid_kid.json
20260629_stage5_detailcontrast_sc125_stage3_div2k_val_gencodec_patch256_n2_fid_kid.json
```

These distribution metrics confirm two things:

```text
1. The balanced condition gate improves FID/KID over the no-extra adapter
   anchor on every strict full552 split without adding payload bits.
2. The counted control stream is not just moving LPIPS/DISTS: it substantially
   improves GenCodec-style patch distribution metrics relative to Stage 3 and
   remains the stronger perceptual/distribution point.
3. The post-control gate improves Kodak/CLIC distribution metrics and all-split
   LPIPS/DISTS, but slightly worsens DIV2K FID/KID and fidelity. This motivates
   a control-aware gate, not another fixed scale sweep.
```

## Research Consequence

Current bottleneck:

```text
CoSER Stage 5 can now produce a much more realistic distribution than Stage 3,
but still loses substantial fidelity and remains behind official CoD-Lite
perceptual quality.
```

Next mainline actions:

```text
1. Evaluate the true control-aware gates below on full552 before promoting any
   post-control gate.
2. Learn a deterministic content gate for condition residual strength from
   decoded CoSER features.
3. Train gates against image/perceptual/fidelity objectives, not a fixed RGB
   blend.
4. Improve the diffusion-control detail head from the same transmitted detail
   payload before adding more counted bits.
5. Build a small curve around no-extra, balanced-gate no-extra, top-k32
   scale=1.0, and top-k32
   perceptual scale=1.25.
6. Keep official CoD-Lite / CoD / RDVQ comparisons clearly labeled and avoid
   external-baseline claims until full curves support them.
```

This preserves the Core MVP: CoSER semantic/detail bitstreams remain the codec
payload, while counted condition-control bits are explicit image-specific
payload and are included in `actual_payload_bpp`.

## 2026-06-29 Handoff: Control-Aware Gate Status

A true control-aware balanced post-control gate was trained and evaluated after
the counted sparse top-k32 condition-control stream:

```text
checkpoint:
  checkpoints/stage4_cod_lite_condition_gate/20260629_stage5_post_control_condition_gate_controlaware_balanced_ft800_b2ga2.pt

full552 eval:
  results/stage4_cod_lite_adapter_eval/20260629_stage5_detailcontrast_hf_g32basis_postgate_controlaware_balanced_full552_affinebasis_ag16_as1_bg32_bs8_k32_b4_topkc256_gr1_br0.25_r0.702655_sc1_huffman_mu_law_mu16_topk_c256_k32_p99_b4_mulaw16/summary.json

actual_payload_bpp:
  0.015486

metrics:
  PSNR     21.1824
  MS-SSIM   0.7151
  LPIPS     0.3889
  DISTS     0.2703
  condition_l1 0.4015
  post_control_gate_mean 0.9969
```

Delta versus the counted-control-only `control_scale=1.0` point:

```text
PSNR     +0.0208
MS-SSIM  +0.0002
LPIPS    -0.0002
DISTS    +0.0012
condition_l1 -0.0003
```

Split deltas show the same pattern on Kodak, CLIC2020 test, and DIV2K val:
slightly better fidelity / condition L1, but consistently worse DISTS. This is
a useful stability result, but it is not a perceptual Stage 5 promotion.

A perceptual-tilted true control-aware gate was then trained as the next
evaluation candidate:

```text
checkpoint:
  checkpoints/stage4_cod_lite_condition_gate/20260629_stage5_post_control_condition_gate_controlaware_perceptual_ft1600_b4.pt

training summary:
  results/stage4_cod_lite_condition_gate/20260629_stage5_post_control_condition_gate_controlaware_perceptual_ft1600_b4/summary.json

training config:
  batch_size 4
  max_steps 1600
  counted_control_mode condition_residual_affine_basis
  control_huffman_key topk_c256_k32_p99_b4_mulaw16
  gate_min 0.0
  gate_max 1.40
  init_gate 1.05
  gate_mean_target 1.05
```

Training-cache summary:

```text
condition_l1_mean 0.3890
pre_control_condition_l1_mean 0.3949
ungated_condition_l1_mean 0.3880
stage4_psnr_mean 21.0116
stage4_ms_ssim_mean 0.7204
lpips_mean 0.3570
dists_loss_mean 0.2389
gate_mean_mean 1.0594
gate_min_mean 0.8720
gate_max_mean 1.2681
gate_std_mean 0.0659
control_payload_bytes_mean 48.7130
```

Interpretation:

```text
The perceptual gate is not yet a result; it is a clean full552 evaluation
candidate. It intentionally lets the post-control residual gate move above 1.0
more often than the balanced gate, while retaining counted-control simulation
inside training. Promotion requires full552/split metrics and, if competitive,
saved reconstructions with GenCodec-style patch-FID/KID.
```

Next exact evaluation command:

```bash
.venv/bin/python scripts/eval_stage4_cod_lite_adapter.py \
  --checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8.pt \
  --post-control-condition-gate-checkpoint checkpoints/stage4_cod_lite_condition_gate/20260629_stage5_post_control_condition_gate_controlaware_perceptual_ft1600_b4.pt \
  --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl \
  --per-image-metrics results/stage4_cod_lite_adapter_eval/20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8_full552_eval/per_image_metrics.jsonl \
  --run-name 20260629_stage5_detailcontrast_hf_g32basis_postgate_controlaware_perceptual_full552_affinebasis_ag16_as1_bg32_bs8_k32_b4_topkc256_gr1_br0.25_r0.702655_sc1_huffman_mu_law_mu16_topk_c256_k32_p99_b4_mulaw16 \
  --crop-size 512 --batch-size 4 --num-workers 4 --blend-alpha 1.0 \
  --counted-control-mode condition_residual_affine_basis \
  --control-basis outputs/stage5_control_basis/20260629_detailcontrast_hf_postaffine_basis_g32s8_k256_train8192_b12/control_basis.pt \
  --control-basis-components 32 --control-basis-candidate-components 256 --control-basis-selection topk \
  --control-basis-range-mode global --control-basis-range-floor 0.000001 \
  --control-codec huffman --control-huffman-key topk_c256_k32_p99_b4_mulaw16 \
  --control-quantizer mu_law --control-mu 16 --control-bits 4 --control-range 0.7026548385620117 \
  --control-groups 32 --control-grid-size 8 --control-affine-groups 16 --control-affine-grid-size 1 \
  --control-affine-gain-range 1.0 --control-affine-bias-range 0.25 --control-scale 1.0 \
  --wandb-mode offline
```
