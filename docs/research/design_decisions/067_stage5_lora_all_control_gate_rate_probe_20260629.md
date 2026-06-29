# Stage 5 LoRA-All Control/Gate and Rate Checkpoint Probe

Date: 2026-06-29 JST

## Decision

Promote the `CoD_Lite_bpp_0_0156` denoiser-all LoRA branch as the current
mainline Stage 5 adapter/control continuation.

Do not promote the `CoD_Lite_bpp_0_0078` branch. It is worse on fidelity,
perceptual metrics, and condition recovery in the first full552 probe.

Use the k16 sparse counted condition-control point as the most efficient
current control payload. k32 gives almost no additional perceptual gain over
k16 at the same backbone/adapter state.

Keep the learned global post-control condition gate as a balanced operating
point. It is a decoder-side deterministic condition-space gate, not RGB
postprocessing, and does not add payload bits.

## Mainline Context

This probe follows the Core MVP rule:

```text
semantic/detail actual bitstream
+ official CoD-Lite diffusion backbone
+ CoSER-owned adapter/control/gate
```

The experiments do not transmit CoD-Lite native bitstreams and do not count
model weights as per-image payload. Any per-image condition-control stream is
included in `actual_payload_bpp`.

## Full552 Results

Evaluation protocol:

```text
Kodak24 + CLIC2020 test428 + DIV2K val100
512 x 512 center crop
actual payload bpp from Stage 3 compress/decompress payloads
```

### No Extra Payload

```text
stage4_noextra anchor:
  run: 20260629_stage4_detailtarget_perceptual_long_ft1800_b8_full552_eval
  bpp: 0.013999
  PSNR: 21.3043
  MS-SSIM: 0.7166
  LPIPS: 0.4080
  DISTS: 0.2780
  condition_l1: 0.4104

denoiser_all LoRA, CoD-Lite 0.0156:
  run: 20260629_stage5_detailtarget_lora0156_denoiser_all_r4_balanced_ft900_full552_eval
  bpp: 0.013999
  PSNR: 21.2926
  MS-SSIM: 0.7149
  LPIPS: 0.3867
  DISTS: 0.2686
  condition_l1: 0.4113
```

Interpretation:

```text
Denoiser-all LoRA gives a real no-extra-bit perceptual improvement over the
current Stage 4 anchor, at a small PSNR/MS-SSIM cost. It should be treated as
the active Stage 5 adapter backbone state, not as an external-baseline win.
```

### k16/k32 Counted Control

Using the existing post-affine sparse top-k basis:

```text
denoiser_all LoRA + k16 sparse control:
  run: 20260629_stage5_lora0156_all_oldbasis_full552_affinebasis_k16
  bpp: 0.014978
  control_bpp: 0.000978
  PSNR: 21.1966
  MS-SSIM: 0.7117
  LPIPS: 0.3665
  DISTS: 0.2575
  condition_l1: 0.4030

denoiser_all LoRA + k32 sparse control:
  run: 20260629_stage5_lora0156_all_oldbasis_full552_affinebasis_k32
  bpp: 0.015484
  control_bpp: 0.001485
  PSNR: 21.1990
  MS-SSIM: 0.7117
  LPIPS: 0.3658
  DISTS: 0.2572
  condition_l1: 0.4024
```

Interpretation:

```text
k16 captures nearly all of the current control benefit.
k32 spends another 0.0005 bpp for only tiny LPIPS/DISTS gains.
Use k16 as the efficient Stage 5 control point unless a better basis/control
representation is introduced.
```

### Fixed Scale Diagnostic and Learned Global Gate

```text
denoiser_all LoRA + k16 control, fixed scale 0.5:
  run: 20260629_stage5_lora0156_all_oldbasis_full552_affinebasis_k16_scale05
  bpp: 0.014978
  control_bpp: 0.000979
  PSNR: 21.2552
  MS-SSIM: 0.7136
  LPIPS: 0.3753
  DISTS: 0.2625
  condition_l1: 0.4048

denoiser_all LoRA + k16 control + learned global condition gate:
  run: 20260629_stage5_lora0156_all_k16_global_gate_perceptual_full552
  bpp: 0.014978
  control_bpp: 0.000978
  PSNR: 21.2573
  MS-SSIM: 0.7142
  LPIPS: 0.3723
  DISTS: 0.2622
  condition_l1: 0.4037
  post_control_gate_mean: 0.9085
```

Split means for the learned global gate:

```text
Kodak24:
  PSNR 20.9136 / MS-SSIM 0.6949 / LPIPS 0.3954 / DISTS 0.2669

CLIC2020 test428:
  PSNR 21.6319 / MS-SSIM 0.7268 / LPIPS 0.3628 / DISTS 0.2594

DIV2K val100:
  PSNR 19.7366 / MS-SSIM 0.6651 / LPIPS 0.4073 / DISTS 0.2733
```

Gate distribution:

```text
q00 0.8203
q05 0.8516
q25 0.8828
q50 0.9062
q75 0.9336
q95 0.9688
q100 1.0234
```

Interpretation:

```text
The learned global gate slightly dominates the fixed scale-0.5 diagnostic
across all aggregate metrics and gives a better balanced point than fixed
scale 1.0. However, the gate distribution is still narrow and high. It is
useful as a safe deterministic condition-space gate, but it is not yet the
large content-adaptive breakthrough.
```

## Rate Checkpoint Probe

The lower-rate `CoD_Lite_bpp_0_0078` backbone was tested with the same
denoiser-all LoRA recipe for 700 steps.

```text
run:
  20260629_stage5_detailtarget_lora0078_denoiser_all_r4_balanced_ft700_full552_eval

bpp:
  0.013999

metrics:
  PSNR: 20.8498
  MS-SSIM: 0.6885
  LPIPS: 0.4001
  DISTS: 0.2785
  condition_l1: 0.4619
```

Interpretation:

```text
The 0.0078 backbone does not provide a useful stronger prior for CoSER at the
current condition quality. It degrades condition recovery and final images.
Do not continue this branch unless a much stronger adapter/control target is
available.
```

## External Baseline Status

This is still not an external-baseline win.

Official CoD-Lite Kodak512 at nominal 0.015625 reports much stronger Kodak
perceptual metrics:

```text
PSNR 20.7667
MS-SSIM 0.7090
LPIPS 0.2259
DISTS 0.1402
FID 38.4785
```

Current CoSER Kodak24, learned global gate point:

```text
bpp about 0.0151
PSNR 20.9136
MS-SSIM 0.6949
LPIPS 0.3954
DISTS 0.2669
```

CoSER is preserving more fidelity than some generative settings, but the
perceptual gap to official CoD-Lite remains very large.

## Next Steps

1. Continue from `CoD_Lite_bpp_0_0156` denoiser-all LoRA, not 0.0078.
2. Treat k16 sparse control as the efficient counted-control point.
3. Use the learned global gate as the balanced operating point, but do not
   keep sweeping gate-only settings as the main research path.
4. Improve the condition/control representation itself:
   diffusion-friendly detail head, better sparse residual basis, or a tiny
   counted control stream with stronger semantic/detail coupling.
5. Consider a controlled `CoD_Lite_bpp_0_0312` probe later, but it is lower
   priority than improving the 0.0156 adapter/control path.
