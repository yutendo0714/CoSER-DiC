# 063 Condition Statistics Ablation and Stats-Loss Probe

Date: 2026-06-29 JST

## Decision

Do not promote condition-statistics-only finetuning as the next Stage 4/5
mainline improvement.

Keep the current no-extra-bit adapter anchor:

```text
20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8_full552_eval
actual_payload_bpp 0.013999
PSNR     21.2351
MS-SSIM   0.7166
LPIPS     0.4139
DISTS     0.2821
condition_l1 0.4108
```

The new stats-loss runs are useful diagnostics:

```text
strong stats/spectrum:
  better MS-SSIM and DISTS
  worse PSNR, LPIPS, and condition L1

light stats/spectrum:
  better PSNR and MS-SSIM
  worse LPIPS and DISTS
```

Therefore the mainline should move from condition-statistics-only tuning to
stronger decoded-bitstream usage:

```text
1. make the detail payload produce a diffusion-control feature that the adapter
   actually depends on;
2. keep channel-statistics and spectrum losses as auxiliary regularizers;
3. do not use them as the main route to perceptual curve improvement.
```

## Full552 CoSER Feature Ablation

Condition-statistics ablations were run on the current no-extra-bit Stage 4
anchor:

```text
checkpoint:
  checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8.pt

manifest:
  results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl
```

| run | pred L1 | delta vs normal | relative L2 | cosine | pred std | highfreq ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| normal | 0.410754 | +0.000000 | 0.669649 | 0.757270 | 0.741719 | 0.240044 |
| semantic latent zero | 0.411963 | +0.001209 | 0.671404 | 0.753754 | 0.731448 | 0.242628 |
| semantic latent shuffle | 0.414842 | +0.004088 | 0.675594 | 0.747794 | 0.717798 | 0.244644 |
| detail context zero | 0.411333 | +0.000579 | 0.670593 | 0.756132 | 0.740145 | 0.241254 |
| detail context shuffle | 0.411600 | +0.000845 | 0.670989 | 0.755814 | 0.739959 | 0.240964 |

Interpretation:

```text
Semantic latent is genuinely used. Shuffling it hurts condition recovery more
than zeroing it, which means image-aligned semantic features matter.

Detail context is still under-used. Zeroing/shuffling detail context hurts only
slightly, even though the Stage 3 detail stream costs about 0.005 bpp and
should be valuable for diffusion control.
```

This supports the earlier diagnosis: the bottleneck is not only condition loss
weighting. The adapter must be redesigned or trained so that the transmitted
detail payload becomes a stronger diffusion-control signal.

Artifacts:

```text
results/stage4_cod_lite_condition_stats/20260629_stage4_detailcontrast_condition_stats_full552/
results/stage4_cod_lite_condition_stats/20260629_stage4_detailcontrast_condition_stats_full552_semzero/
results/stage4_cod_lite_condition_stats/20260629_stage4_detailcontrast_condition_stats_full552_semshuffle/
results/stage4_cod_lite_condition_stats/20260629_stage4_detailcontrast_condition_stats_full552_detailzero/
results/stage4_cod_lite_condition_stats/20260629_stage4_detailcontrast_condition_stats_full552_detailshuffle/
```

## Stats/Spectrum Finetune Probes

Two finetunes were run from the current no-extra-bit anchor.

### Strong Stats/Spectrum

```text
run:
  20260629_stage4_detailcontrast_stats_spectrum_ft800_b8

checkpoint:
  checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailcontrast_stats_spectrum_ft800_b8.pt

full552 eval:
  results/stage4_cod_lite_adapter_eval/20260629_stage4_detailcontrast_stats_spectrum_ft800_b8_full552_eval/summary.json
```

Delta versus the current anchor:

```text
bpp      +0.000000
PSNR     -0.0114
MS-SSIM  +0.0017
LPIPS    +0.0029
DISTS    -0.0009
condition_l1 +0.0056
```

Condition-statistics effect:

```text
pred std:
  0.7417 -> 0.7557

pred highfreq ratio:
  0.2400 -> 0.2468

target std / highfreq:
  0.7954 / 0.2575
```

It successfully moves the predicted condition distribution toward the native
CoD-Lite condition, especially in high-frequency ratio, but the final image
tradeoff is mixed and LPIPS gets worse.

### Light Stats/Spectrum

```text
run:
  20260629_stage4_detailcontrast_statslite_ft500_b8

checkpoint:
  checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailcontrast_statslite_ft500_b8.pt

full552 eval:
  results/stage4_cod_lite_adapter_eval/20260629_stage4_detailcontrast_statslite_ft500_b8_full552_eval/summary.json
```

Delta versus the current anchor:

```text
bpp      +0.000000
PSNR     +0.0472
MS-SSIM  +0.0012
LPIPS    +0.0051
DISTS    +0.0033
condition_l1 +0.0011
```

Split deltas:

```text
Kodak24:
  PSNR +0.0536 / MS-SSIM +0.0018 / LPIPS +0.0100 / DISTS +0.0051

CLIC2020 test428:
  PSNR +0.0449 / MS-SSIM +0.0011 / LPIPS +0.0044 / DISTS +0.0030

DIV2K val100:
  PSNR +0.0555 / MS-SSIM +0.0012 / LPIPS +0.0065 / DISTS +0.0041
```

The light stats run is a fidelity-biased curve point, not a perceptual
improvement. It may be useful later when building rate-perception-faithfulness
curves, but it should not replace the current perceptual/control path.

## Mainline Consequence

Condition statistics and spectrum matching are real but secondary.

Next work should focus on:

```text
1. detail-aware adapter training where zero/shuffled detail must hurt;
2. a stronger diffusion-control detail projection from the same transmitted
   detail payload;
3. counted-control/gate training that uses condition statistics as a guard,
   not as the primary objective;
4. Stage3 rate allocation experiments where detail bits are optimized for
   diffusion control rather than only RGB reconstruction.
```

Promotion rule:

```text
Do not promote a Stage 4/5 candidate if it only improves PSNR/MS-SSIM while
regressing LPIPS/DISTS, unless it is explicitly labeled as a fidelity curve
point.
```
