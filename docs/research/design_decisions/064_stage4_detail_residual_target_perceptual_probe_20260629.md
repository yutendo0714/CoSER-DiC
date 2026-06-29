# 064 Stage 4 Detail-Residual Target and Perceptual Probe

Date: 2026-06-29

## Decision

Add `--detail-residual-target-weight` to Stage 4 CoD-Lite adapter training as
mainline infrastructure, and promote
`20260629_stage4_detailtarget_perceptual_ft700_b8` as the current no-extra-bit
Stage 4 adapter candidate.

This is not a Stage 5 result and does not close the external CoD-Lite gap. It
is a small but real mainline improvement because it:

- keeps `actual_payload_bpp` unchanged at `0.013999`
- improves PSNR, LPIPS, DISTS, and condition L1 versus the previous Stage 4
  detail-contrast anchor
- makes the decoded detail payload measurably more useful in condition-space
  and image-space ablations

MS-SSIM is nearly flat but slightly worse, so this should be treated as an
intermediate candidate rather than final promotion.

## Why

The full552 condition-statistics ablation in decision 063 showed:

- semantic latent is meaningfully used
- detail context is still weakly used
- stats/spectrum-only tuning is auxiliary and does not directly solve the
  perceptual gap

The missing pressure was direct: the model was asked to make the final
condition better, but not explicitly asked for the detail branch to explain the
condition residual. The new loss compares:

```text
pred_detail_residual = pred_cond - pred_cond_with_zero_detail
target_condition_residual = target_cond - base_cond
```

and adds an L1 penalty between those tensors. This is training-only supervision
from the reference image. At inference, the decoder still uses only decoded
CoSER tensors, fixed adapter weights, and fixed CoD-Lite weights. No
image-specific side information is introduced.

## Implementation

Updated:

- `scripts/train_stage4_cod_lite_adapter.py`

New CLI:

```bash
--detail-residual-target-weight
```

The implementation also logs:

- `detail_residual_target_loss`
- `detail_residual_target_pred_l1`
- `detail_residual_target_target_l1`

Verification:

```bash
.venv/bin/python -m py_compile scripts/train_stage4_cod_lite_adapter.py
```

## Runs

### Condition-target-heavy probe

Run:

```text
20260629_stage4_detailtarget_ft700_b8
```

Full552 eval:

```text
results/stage4_cod_lite_adapter_eval/20260629_stage4_detailtarget_ft700_b8_full552_eval/summary.json
```

Outcome versus previous anchor:

| metric | anchor | detailtarget | delta |
|---|---:|---:|---:|
| actual_payload_bpp | 0.013999 | 0.013999 | +0.000000 |
| PSNR | 21.2351 | 21.3045 | +0.0694 |
| MS-SSIM | 0.7166 | 0.7176 | +0.0010 |
| LPIPS-Alex | 0.4139 | 0.4166 | +0.0027 |
| DISTS | 0.2821 | 0.2847 | +0.0026 |
| condition L1 | 0.4108 | 0.4096 | -0.0012 |

Interpretation:

This run increased detail dependence and improved fidelity/condition L1, but
it worsened perceptual metrics. It should not be promoted.

### Perceptual-balanced detail-target probe

Run:

```text
20260629_stage4_detailtarget_perceptual_ft700_b8
```

Full552 eval:

```text
results/stage4_cod_lite_adapter_eval/20260629_stage4_detailtarget_perceptual_ft700_b8_full552_eval/summary.json
```

Outcome versus previous anchor:

| metric | anchor | detailtarget_perceptual | delta |
|---|---:|---:|---:|
| actual_payload_bpp | 0.013999 | 0.013999 | +0.000000 |
| PSNR | 21.2351 | 21.2816 | +0.0465 |
| MS-SSIM | 0.7166 | 0.7165 | -0.0001 |
| LPIPS-Alex | 0.4139 | 0.4110 | -0.0029 |
| DISTS | 0.2821 | 0.2795 | -0.0026 |
| condition L1 | 0.4108 | 0.4102 | -0.0006 |

Interpretation:

This is a better no-extra-bit Stage 4 candidate than the previous anchor, but
the gain is still incremental and far from external-baseline competitiveness.

## Detail-Use Ablation

Condition-space ablation:

| run | normal L1 | zero-detail penalty | shuffle-detail penalty | pred HF | target HF |
|---|---:|---:|---:|---:|---:|
| old anchor | 0.410754 | +0.000579 | +0.000845 | 0.2400 | 0.2575 |
| detailtarget | 0.409599 | +0.001704 | +0.001692 | 0.2415 | 0.2575 |
| detailtarget_perceptual | 0.410228 | +0.001530 | +0.001621 | 0.2436 | 0.2575 |

Image-space ablation for `detailtarget_perceptual`:

| ablation | PSNR delta | MS-SSIM delta | LPIPS delta | DISTS delta | condition L1 delta |
|---|---:|---:|---:|---:|---:|
| zero detail | +0.0274 | +0.0000 | +0.0057 | +0.0035 | +0.0015 |
| shuffled detail | +0.0006 | -0.0004 | +0.0035 | +0.0020 | +0.0016 |

Interpretation:

The detail branch is now doing perceptual work: removing or shuffling decoded
detail makes LPIPS/DISTS and condition L1 worse, while sometimes improving PSNR.
That is the expected rate-perception-fidelity tradeoff for a diffusion-control
detail path. The effect is still small, so the next step is to amplify this
without losing structure.

## Current Candidate Ranking

Use this for the next no-extra-bit Stage 4 starting point:

```text
checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailtarget_perceptual_ft700_b8.pt
```

Keep this as a fidelity-biased diagnostic:

```text
checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailtarget_ft700_b8.pt
```

Keep the old anchor for reproducibility:

```text
checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8.pt
```

## Next Mainline Move

Do not return to fixed alpha sweeps or RGB postprocessing.

Next priority:

1. Continue from `detailtarget_perceptual`.
2. Add a deterministic content/condition gate trained to preserve the
   `detailtarget_perceptual` perceptual gain while recovering MS-SSIM.
3. Test a longer run with the same loss family on a larger train cache.
4. If the adapter saturates, test low-LR LoRA/partial unfreeze while keeping
   CoSER semantic/detail payload as the only transmitted no-control stream.
5. Build additional actual-payload operating points; this single-point gain is
   not enough for BD-rate claims.

## Bottom Line

This is the first small result in this sequence that moves in the correct
mainline direction:

```text
same actual payload
+ better condition recovery
+ better LPIPS/DISTS
+ better PSNR
+ measurable use of decoded detail payload
```

It is still a long way from CoD-Lite/CoD/RDVQ/StableCodec curve wins. Treat it
as a better Stage 4 foothold, not as proof of Stage 5 success.
