# Stage 5 Detail-Control Fusion Probe

Date: 2026-06-29 JST

## Question

Can the same transmitted Stage 3 detail payload guide the CoD-Lite diffusion
condition more directly, without adding bits or relying on RGB postprocessing?

## Implementation

Added a deeper decoder-side detail-control path to
`CoSERToCoDLiteConditionPyramidAdapter`:

```text
detail_context -> detail_proj -> detail_blocks
  -> optional concat with base-condition features
  -> detail_control_blocks
  -> zero-init condition-space delta
```

New knobs:

```text
--detail-control-blocks
--detail-control-condition-fusion
```

The branch uses only decoded CoSER detail features already available at the
decoder. It does not add side information, so actual_payload_bpp is unchanged
except for the existing counted k16 control stream used in Stage 5.

Verification:

```text
.venv/bin/python -m py_compile src/coserdic/models/gencodec_backbone.py \
  scripts/train_stage4_cod_lite_adapter.py \
  scripts/sweep_stage5_lora_backbone.py \
  scripts/eval_stage4_cod_lite_adapter.py

.venv/bin/python -m pytest tests/test_gencodec_backbone.py -q
24 passed
```

## Runs

### Balanced detail-control fusion

Training:

```text
20260629_stage5_detailcontrol_fusion_lora0156_all_k16_ft600_b4ga2
```

Full552 evaluation:

```text
20260629_stage5_detailcontrol_fusion_lora0156_all_k16_ft600_full552_eval

actual_payload_bpp  0.014977
PSNR                21.2634
MS-SSIM             0.7137
LPIPS               0.3649
DISTS               0.2597
condition_l1        0.4014
```

Compared with the previous old k16 full552 anchor:

```text
old k16:
  bpp      0.014978
  PSNR     21.1966
  MS-SSIM  0.7117
  LPIPS    0.3665
  DISTS    0.2575

detail-control fusion:
  bpp      0.014977
  PSNR     21.2634
  MS-SSIM  0.7137
  LPIPS    0.3649
  DISTS    0.2597
```

Interpretation:

```text
The new branch improves fidelity and LPIPS slightly at the same actual payload
bpp, but DISTS regresses. This is a real but small mainline gain, not a Stage 5
external-baseline win.
```

### DISTS-heavy detail-control fusion

Training:

```text
20260629_stage5_detailcontrol_fusion_lora0156_all_k16_dists_ft500_b4ga2
```

Full552 evaluation:

```text
20260629_stage5_detailcontrol_fusion_lora0156_all_k16_dists_ft500_full552_eval

actual_payload_bpp  0.014978
PSNR                21.3102
MS-SSIM             0.7158
LPIPS               0.3682
DISTS               0.2630
condition_l1        0.4014
```

Interpretation:

```text
DISTS-heavy loss improves fidelity metrics further, but worsens LPIPS/DISTS.
This suggests the bottleneck is not simple DISTS reweighting. The current detail
payload is still more useful for fidelity/condition stabilization than for
perceptual structure matching.
```

## Decision

Do not promote the DISTS-heavy branch.

Keep the balanced detail-control fusion implementation as a useful mainline
adapter option, but do not treat it as a completed Stage 5 result.

Next work should change the information flow more substantially:

```text
1. diffusion-friendly detail payload/head from the same transmitted residual
2. Stage4-aware detail allocation and detail decoder training
3. semantic/detail bpp allocation sweeps evaluated after CoD-Lite decoding
4. stronger condition-space detail losses only after the payload carries the
   right information
```

The important lesson is that adding a condition-aware detail-control head helps
slightly, but the curve gap to official CoD-Lite is too large for adapter/loss
tuning alone.
