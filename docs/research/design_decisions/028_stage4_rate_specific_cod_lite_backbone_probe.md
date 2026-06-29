# Stage 4 Rate-Specific CoD-Lite Backbone Probe

Date: 2026-06-29

## Decision

Keep the detail-aware CoSER-to-CoD-Lite condition adapter as the Stage 4
mainline, but treat the fixed CoD-Lite decoder backbone checkpoint as an
experimental axis.

This is still a CoSER codec:

```text
transmitted bits:
  CoSER semantic VQ stream
  CoSER detail residual stream

not transmitted:
  fixed CoD-Lite backbone weights
  fixed CoSER adapter weights
  fixed decoder-side configuration
```

The paper bpp remains the Stage 3 `actual_payload_bpp`, because no image-specific
CoD-Lite bitstream or prompt/control map is transmitted.

## Why

The detail-aware adapter results showed that decoded CoSER features can drive a
frozen official CoD-Lite one-step backbone, but the best operating point depends
on the pretrained decoder prior:

```text
bpp0.0156 checkpoint:
  better current LPIPS / FID anchor

bpp0.0312 checkpoint:
  better current PSNR / MS-SSIM / DISTS anchor

bpp0.0078 checkpoint:
  not promoted from the first probe
```

This is useful mainline evidence. It changes the diffusion prior used by the
condition adapter rather than blending final RGB outputs.

## Implementation Note

The CoD-Lite bpp0.0039 / bpp0.0078 checkpoints use an upsampled condition path.
The official `calculate_indices_size()` value is the compressed index-grid size,
not the actual native condition tensor shape consumed by the decoder.

Fix:

```text
CoDLiteOneStepBackbone.condition_size()
  now caches the true native condition shape from native_condition(dummy_rgb)
```

This keeps the adapter output shape aligned with the official backbone for both
ordinary and up2x rate checkpoints.

## Probe Runs

Shared Stage 3 payload:

```text
strict split:
  Kodak24 + CLIC2020 test428 + DIV2K val100

actual_payload_bpp:
  0.013999
```

### bpp0.0078 Transfer Probe

```text
run:
  20260628_stage4_detailaware_adapter_bpp0078_transfer_ft600_b2

checkpoint:
  checkpoints/stage4_cod_lite_adapter/
  20260628_stage4_detailaware_adapter_bpp0078_transfer_ft600_b2.pt

eval:
  20260628_stage4_detailaware_adapter_bpp0078_transfer_ft600_b2_limit64_eval
```

Limit64 result:

| model | PSNR | MS-SSIM | LPIPS | DISTS | condition_l1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage 3 | 22.3652 | 0.7568 | 0.5450 | 0.3497 | n/a |
| bpp0.0078 adapter | 21.2619 | 0.7134 | 0.4225 | 0.2982 | 0.4499 |

Judgment:

```text
not promoted
too much fidelity loss relative to the bpp0.0156 / bpp0.0312 paths
```

### bpp0.0312 Transfer Probe

```text
run:
  20260628_stage4_detailaware_adapter_bpp0312_transfer_ft600_b2

checkpoint:
  checkpoints/stage4_cod_lite_adapter/
  20260628_stage4_detailaware_adapter_bpp0312_transfer_ft600_b2.pt

full552 eval:
  20260628_stage4_detailaware_adapter_bpp0312_transfer_ft600_b2_full552_eval

saved full552 eval:
  20260628_stage4_detailaware_adapter_bpp0312_transfer_ft600_b2_full552_save_eval
```

Full552 result:

| model | PSNR | MS-SSIM | LPIPS | DISTS | condition_l1 | base_condition_l1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage 3 | 21.9951 | 0.7348 | 0.5758 | 0.3536 | n/a | n/a |
| bpp0.0156 ft600 | 21.2840 | 0.7155 | 0.4358 | 0.3011 | 0.4162 | 0.5372 |
| bpp0.0156 ft3000 | 21.3638 | 0.7182 | 0.4384 | 0.3023 | 0.4167 | 0.5372 |
| bpp0.0312 transfer | 21.4438 | 0.7216 | 0.4402 | 0.2917 | 0.4589 | 0.5992 |

Split-wise bpp0.0312 result:

| split | PSNR | MS-SSIM | LPIPS | DISTS | dPSNR vs Stage3 | dLPIPS vs Stage3 | dDISTS vs Stage3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Kodak24 | 21.1635 | 0.7050 | 0.4827 | 0.3126 | -0.5039 | -0.1539 | -0.0606 |
| CLIC2020 test428 | 21.8195 | 0.7337 | 0.4297 | 0.2884 | -0.5443 | -0.1312 | -0.0615 |
| DIV2K val100 | 19.9032 | 0.6742 | 0.4749 | 0.3007 | -0.5924 | -0.1495 | -0.0641 |

Patch-FID / KID, GenCodec-style protocol:

| dataset | images | patch size | patches | FID | KID mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kodak | 24 | 64 | 2712 | 104.7621 | 0.036518 |
| CLIC2020 test | 428 | 256 | 2140 | 87.1151 | 0.025608 |
| DIV2K val | 100 | 256 | 500 | 182.9174 | 0.038770 |

Compared with bpp0.0156 ft600:

```text
improves:
  PSNR
  MS-SSIM
  DISTS
  DISTS win rate

regresses:
  LPIPS
  Kodak / CLIC / DIV2K patch-FID
  CLIC / DIV2K KID
```

Judgment:

```text
bpp0.0312 transfer becomes the current DISTS/fidelity-leaning raw Stage 4
candidate, but it does not replace ft600 as the perceptual/distribution anchor.
```

## Current Anchors

Use the following anchors for next Stage 4 work:

```text
bpp0.0156 ft600:
  best current raw LPIPS anchor
  better current patch-FID/KID anchor

bpp0.0156 ft3000:
  continuation check showing longer training improves fidelity but not LPIPS

bpp0.0312 transfer:
  best current raw DISTS/fidelity anchor
  worse patch-FID/KID than ft600
```

None of these is a Stage 5 result.

## Next Work

1. Train a stronger condition-imitation adapter on a larger clean cache.
2. Add a deterministic condition-space gate with explicit anti-collapse terms.
3. Keep both bpp0.0156 and bpp0.0312 backbone tracks active until one clearly
   dominates on full552 and visual audit.
4. Do not spend more time on fixed RGB alpha as a method.
5. After adapter saturation, test low-LR partial unfreeze or LoRA.
