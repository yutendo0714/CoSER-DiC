# 017 Stage 3 Fixed Decoder Unsharp Postprocess

Date: 2026-06-27 JST
Status: Historical Stage 3 decision; Stage 4 direction is now superseded by
`021_stage4_cod_codlite_parallel_backbone_policy.md`

## Decision

Add fixed decoder-side postprocess presets to Stage 3 evaluation, but do not
replace the neutral anchor yet.

The current safe perceptual preset is:

```text
decoder_postprocess: unsharp3x3
decoder_postprocess_strength: 2.0
```

A stronger metric-max diagnostic preset is:

```text
decoder_postprocess: unsharp3x3
decoder_postprocess_strength: 4.0
```

Both improve LPIPS/DISTS at unchanged `actual_payload_bpp`, but they are decoder
configuration choices, not image-specific side information. If a future encoder
selects the postprocess mode or strength per image, that control signal must be
transmitted and counted in `actual_payload_bpp`.

## Implementation

Added:

```text
src/coserdic/models/postprocess.py
tests/test_decoder_postprocess.py
```

Updated:

```text
scripts/eval_stage3_uniform_residual_bitstream.py
```

New options:

```bash
--decoder-postprocess {none,gaussian3x3,unsharp3x3}
--decoder-postprocess-strength FLOAT
```

The postprocess is applied after:

```text
x_sem + bilinear_upsampled_residual_grid
```

It never changes the entropy-coded semantic/detail payload streams. Non-default
settings are recorded in `entropy_model_version` only for debug/audit metadata.

## Kodak Probe

Protocol:

```text
Kodak 24
crop: 256x256
detail: b5 semantic-position-left context Huffman
actual_payload_bpp: 0.018473 for all rows
```

| mode | strength | PSNR | MS-SSIM | LPIPS | DISTS |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 0.0 | 21.131926 | 0.681533 | 0.736780 | 0.424619 |
| gaussian3x3 | 0.25 | 21.133600 | 0.681460 | 0.738346 | 0.425793 |
| gaussian3x3 | 0.50 | 21.135021 | 0.681380 | 0.739869 | 0.426864 |
| unsharp3x3 | 0.25 | 21.130000 | 0.681601 | 0.735179 | 0.423400 |
| unsharp3x3 | 0.50 | 21.127821 | 0.681663 | 0.733547 | 0.422117 |
| unsharp3x3 | 0.75 | 21.125391 | 0.681720 | 0.731878 | 0.420781 |
| unsharp3x3 | 1.00 | 21.122711 | 0.681771 | 0.730160 | 0.419460 |
| unsharp3x3 | 1.50 | 21.116604 | 0.681855 | 0.726609 | 0.416798 |
| unsharp3x3 | 2.00 | 21.109511 | 0.681918 | 0.722938 | 0.414251 |
| unsharp3x3 | 3.00 | 21.092402 | 0.681979 | 0.715257 | 0.409909 |
| unsharp3x3 | 4.00 | 21.071438 | 0.681954 | 0.707199 | 0.407145 |

Gaussian smoothing is rejected. Unsharp improves Kodak LPIPS/DISTS monotonically
in this probe, while gradually reducing PSNR.

## CoSER Common LIC

Protocol:

```text
Kodak 24 + CLIC Professional Validation 41 + DIV2K validation 100
total: 165 images
crop: 256x256
actual_payload_bpp: 0.018369732481 for all rows
```

| preset | PSNR | MS-SSIM | LPIPS | DISTS |
| --- | ---: | ---: | ---: | ---: |
| none | 21.390054 | 0.693741 | 0.694714 | 0.409676 |
| unsharp3x3 0.50 | 21.381327 | 0.693837 | 0.691099 | 0.407176 |
| unsharp3x3 1.00 | 21.370798 | 0.693907 | 0.687402 | 0.404535 |
| unsharp3x3 2.00 | 21.344658 | 0.693968 | 0.679759 | 0.399230 |
| unsharp3x3 4.00 | 21.274349 | 0.693789 | 0.663952 | 0.391251 |

Per-image wins over the deterministic neutral anchor:

| preset | LPIPS wins | DISTS wins | rate delta |
| --- | ---: | ---: | ---: |
| unsharp3x3 0.50 | 162 / 165 | 164 / 165 | 0.0 |
| unsharp3x3 1.00 | 162 / 165 | 163 / 165 | 0.0 |
| unsharp3x3 2.00 | 163 / 165 | 163 / 165 | 0.0 |
| unsharp3x3 4.00 | 163 / 165 | 158 / 165 | 0.0 |

Artifacts:

```text
results/analysis/stage3_decoder_postprocess/20260627_b5_common_unsharp_summary.csv
results/analysis/stage3_decoder_postprocess/20260627_b5_common_unsharp_metric_max_summary.csv
results/analysis/stage3_decoder_postprocess/20260627_b5_common_unsharp200_vs_none_lpips.json
results/analysis/stage3_decoder_postprocess/20260627_b5_common_unsharp200_vs_none_dists.json
results/analysis/stage3_decoder_postprocess/20260627_b5_common_unsharp400_vs_none_lpips.json
results/analysis/stage3_decoder_postprocess/20260627_b5_common_unsharp400_vs_none_dists.json
```

## Distribution Metrics

Run:

```text
20260627_stage3_b5_semposleft_g4_coser_common_lic_pp_unsharp200_recon_export
wandb/offline-run-20260627_173614-7nfcvxk4
20260627_stage3_b5_semposleft_g4_coser_common_lic_pp_unsharp400_recon_export
wandb/offline-run-20260627_174734-k0kmpz7l
```

Compared to the prior neutral reconstruction export:

| setting | image FID | image KID | patch128 FID | patch128 KID |
| --- | ---: | ---: | ---: | ---: |
| neutral b5 | 285.503783 | 0.084817 | 226.920191 | 0.104232 |
| unsharp3x3 2.0 | 282.247165 | 0.082436 | 222.928785 | 0.107602 |
| unsharp3x3 4.0 | 279.442998 | 0.082808 | 229.295739 | 0.122919 |

Unsharp 2.0 improves image-level FID/KID and patch-level FID, but worsens
patch-level KID. Unsharp 4.0 further improves image-level FID, but worsens
patch-level FID/KID relative to 2.0. Treat FID/KID as secondary evidence and do
not promote 4.0 solely from average LPIPS/DISTS.

## Visual Audit

Worst-case galleries:

```text
results/analysis/stage3_visual_galleries/20260627_stage3_b5_common_unsharp200_worst_lpips.md
results/analysis/stage3_visual_galleries/20260627_stage3_b5_common_unsharp200_worst_dists.md
results/analysis/stage3_visual_galleries/20260627_stage3_b5_common_unsharp200_lowest_psnr.md
results/analysis/stage3_visual_galleries/20260627_stage3_b5_common_unsharp400_worst_lpips.md
results/analysis/stage3_visual_galleries/20260627_stage3_b5_common_unsharp400_worst_dists.md
```

Visual inspection of top LPIPS/DISTS cases shows that the dominant failure mode
is still semantic low-rate blur and missing high-frequency structure, not obvious
new ringing from the fixed unsharp postprocess. The postprocess can sharpen
existing coarse structures, but it cannot recover water, foliage, window grids,
or fine texture lost by the semantic stage.

## Research Takeaway

Fixed unsharp postprocess is a useful no-extra-payload perceptual preset for
Stage 3. It should be reported separately from the neutral anchor:

```text
neutral anchor: decoder_postprocess=none
safer perceptual preset: decoder_postprocess=unsharp3x3, strength=2.0
metric-max diagnostic: decoder_postprocess=unsharp3x3, strength=4.0
```

This result strengthens the case that Stage 4/5 should use a model-side
perceptual reconstruction module. The active version of that idea is no longer
a generic ResUNet refiner; it is the CoD-Lite / CoD diffusion-backbone
integration described in Decision 021. The learned module should not merely
sharpen; it must restore semantically plausible high-frequency structure while
preserving actual-byte stream accounting.
