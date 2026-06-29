# Stage 5 Control-Aware Adapter Training Probe

Date: 2026-06-29

## Question

Can the CoSER-to-CoD-Lite adapter be improved by training it with the same
counted condition-control stream used at evaluation time?

This is a mainline Stage 5 question because the control stream is entropy coded
and counted in `actual_payload_bpp`. It is not RGB postprocessing and does not
replace the CoSER semantic/detail streams.

## Implementation

Added optional control-aware training to:

```text
scripts/train_stage4_cod_lite_adapter.py
```

New mode:

```text
--train-counted-control-mode condition_residual_affine_basis
```

Also added a guard for the raw adapter condition:

```text
--pre-control-condition-l1-weight
```

This keeps the pre-control adapter condition close to the teacher while still
training the post-control condition. The intent is to make the counted control
stream complementary rather than letting it carry the entire correction.

The training loop now can:

1. predict a CoD-Lite condition from decoded CoSER features
2. compute an entropy-coded affine+basis control correction against the teacher
   condition
3. decode that correction immediately
4. train image/condition losses on the corrected condition

The discrete control path is non-differentiable, but the corrected condition is
used by the loss, so the adapter is optimized for the actual control operating
point rather than for a purely continuous condition target.

## Training Run

```text
checkpoint:
  checkpoints/stage4_cod_lite_adapter/20260629_stage5_controlaware_adapter_lora0156_all_k16_ft400_b4ga2.pt

training summary:
  results/stage4_cod_lite_adapter/20260629_stage5_controlaware_adapter_lora0156_all_k16_ft400_b4ga2/summary.json

wandb:
  wandb/offline-run-20260629_210337-wlj0jb4y
```

Training anchor:

```text
init:
  20260629_stage5_detailtarget_lora0156_denoiser_all_r4_balanced_ft900_b4ga2

control:
  condition_residual_affine_basis
  groups 32, grid 8
  affine groups 16, affine grid 1
  topk 16 / candidate 256
  Huffman key topk_c256_k16_p99_b4_mulaw16
```

Training diagnostics:

| metric | value |
| --- | ---: |
| condition_l1_mean | 0.3873 |
| pre_control_condition_l1_mean | 0.3934 |
| control_condition_l1_delta_mean | -0.0061 |
| control_payload_bytes_mean | 32.44 |
| control_payload_bpp_512_mean | 0.000990 |
| LPIPS train mean | 0.3430 |
| DISTS train mean | 0.2331 |
| stage4 PSNR train mean | 21.1485 |

The counted control path is active during training and improves condition L1,
but the improvement is modest.

## Full552 Results

Strict GenCodec-style full552 split:

```text
Kodak24 + CLIC2020 test428 + DIV2K val100
crop: 512 center crop
```

Main counted-control evaluation:

```text
results/stage4_cod_lite_adapter_eval/
  20260629_stage5_controlaware_adapter_lora0156_all_k16_ft400_full552_eval/
```

Saved reconstruction evaluation:

```text
results/stage4_cod_lite_adapter_eval/
  20260629_stage5_controlaware_adapter_lora0156_all_k16_ft400_full552_save_eval/
```

Comparison:

| run | bpp | PSNR | MS-SSIM | LPIPS | DISTS | condition L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA-all no-extra | 0.013999 | 21.2926 | 0.7149 | 0.3867 | 0.2686 | 0.4113 |
| LoRA-all + k16 control | 0.014980 | 21.1959 | 0.7116 | 0.3664 | 0.2574 | 0.4030 |
| control-aware FT400, no control | 0.013999 | 21.3594 | 0.7166 | 0.3888 | 0.2716 | 0.4106 |
| control-aware FT400 + k16 control | 0.014978 | 21.2492 | 0.7130 | 0.3655 | 0.2589 | 0.4017 |
| pre-control guarded FT300 + k16 control | 0.014978 | 21.2452 | 0.7121 | 0.3658 | 0.2591 | 0.4013 |

Interpretation:

```text
The control-aware adapter update improves fidelity without counted control, but
it worsens LPIPS/DISTS. With the k16 counted control stream, LPIPS improves
slightly over the previous k16 control point, while DISTS regresses.

This is not a promoted Stage 5 result.
```

The pre-control guarded run slightly improves condition L1 over FT400 but does
not recover DISTS. It therefore confirms that this simple auxiliary loss is not
enough.

Split metrics for the promoted comparison candidate:

| split | bpp | PSNR | MS-SSIM | LPIPS | DISTS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kodak24 | 0.015092 | 20.9057 | 0.6932 | 0.3889 | 0.2626 |
| CLIC2020 test428 | 0.014725 | 21.6235 | 0.7256 | 0.3563 | 0.2564 |
| DIV2K val100 | 0.016029 | 19.7297 | 0.6639 | 0.3993 | 0.2692 |

## Patch-FID / KID

GenCodec-style protocol:

```text
Kodak: patch64, fid_patch_num=2
CLIC2020 / DIV2K: patch256, fid_patch_num=2
backend: torch-fidelity
```

Artifacts:

```text
results/distribution_metrics/
  20260629_stage5_controlaware_lora0156_all_k16_ft400_kodak_gencodec_patch64_n2_fid_kid.json
  20260629_stage5_controlaware_lora0156_all_k16_ft400_clic_gencodec_patch256_n2_fid_kid.json
  20260629_stage5_controlaware_lora0156_all_k16_ft400_div2k_gencodec_patch256_n2_fid_kid.json
```

| split | FID | KID mean |
| --- | ---: | ---: |
| Kodak24 | 80.9884 | 0.020049 |
| CLIC2020 test428 | 62.7218 | 0.010622 |
| DIV2K val100 | 161.8326 | 0.021273 |

Compared with the previous detailtarget long balanced/postgate distribution
points, this branch is worse on CLIC and DIV2K FID/KID. It is therefore not a
good long-run continuation seed.

## Failure Pattern

Worst PSNR regressions are mostly high-PSNR CLIC images where the diffusion
prior improves LPIPS/DISTS but removes pixel-level fidelity:

```text
worst PSNR delta:
  index 415, CLIC mobile
  PSNR 36.54 -> 32.05
  LPIPS 0.4297 -> 0.2533
  DISTS 0.2703 -> 0.2183
```

This reinforces the current diagnosis:

```text
The adapter/control path can create perceptual gains, but it still does not
know when fidelity should dominate. Training against counted control alone is
not enough.
```

## Decision

Do not promote `20260629_stage5_controlaware_adapter_lora0156_all_k16_ft400_b4ga2`.

Keep the implementation because it is useful infrastructure:

```text
--train-counted-control-mode condition_residual_affine_basis
```

But do not spend a long run on this exact loss recipe.

Next mainline direction:

1. Add a control-aware objective that explicitly protects DISTS/FID, not only
   condition L1 and image losses.
2. Train the adapter to predict a residual that is complementary to the fixed
   k16 control basis instead of letting the discrete correction chase a moving
   target.
3. Add a deterministic content gate or control-strength predictor trained with
   fidelity guards, but keep it in condition/control space.
4. Revisit Stage 3 detail-control heads so the transmitted detail payload is
   more directly useful for CoD-Lite condition recovery.

The current best practical k16 counted-control anchor remains:

```text
results/stage4_cod_lite_adapter_eval/
  20260629_stage5_lora0156_all_lorabasis_full552_affinebasis_k16/
```

The current best distribution-oriented anchors remain the earlier detailtarget
long balanced/postgate saved runs documented in:

```text
docs/research/design_decisions/065_stage4_detailtarget_long_and_stage5_rebased_control_20260629.md
```
