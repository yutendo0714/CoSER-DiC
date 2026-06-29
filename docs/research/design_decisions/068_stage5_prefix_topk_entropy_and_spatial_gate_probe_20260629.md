# Stage 5 Prefix-TopK Entropy and Spatial Gate Probe

Date: 2026-06-29 JST

## Decision

Do not promote the new probes as the primary Stage 5 result.

Keep the implementation support for `prefix_topk` basis control with entropy
payloads because it is a valid counted-control codec path and slightly improves
the k32-style perceptual point at lower bpp than k32. However, the gain is too
small to be the main route to an external-baseline win.

Keep the spatial condition gate checkpoint as a diagnostic / secondary operating
point. It improves LPIPS/DISTS over the global gate but loses fidelity, and it
does not beat the ungated k16 control point on LPIPS/DISTS.

The active mainline remains:

```text
CoD_Lite_bpp_0_0156 denoiser-all LoRA
+ k16 sparse counted condition control
+ better control-aware adapter/gate training
+ later diffusion-friendly detail / entropy allocation changes
```

## What Changed

`scripts/eval_stage4_cod_lite_adapter.py` now supports:

```text
--control-basis-selection prefix_topk
--control-codec huffman
```

The implemented entropy layout is:

```text
prefix values:
  static position Huffman from control_huffman_priors

tail top-k indices:
  fixed-bit local indices over the tail candidate set

tail top-k values:
  static value Huffman from sparse_topk_control_priors
```

This is still an actual transmitted control stream. Its bytes are included in
`control_payload_bpp` and therefore in `actual_payload_bpp`.

The same evaluator now also writes `split_metrics` into `summary.json`, using
`source_path` to classify the strict full552 protocol into:

```text
kodak24
clic2020_test428
div2k_val100
unknown
```

This prevents future Stage 5 promotion decisions from relying only on aggregate
means.

## Full552 Protocol

```text
Kodak24 + CLIC2020 test428 + DIV2K val100
512 x 512 center crop
actual_payload_bpp from Stage 3 payload + counted condition-control payload
checkpoint: CoD_Lite_bpp_0_0156 denoiser-all LoRA
```

## Aggregate Results

```text
all_noextra:
  bpp 0.013999
  PSNR 21.2926 / MS-SSIM 0.7149
  LPIPS 0.3867 / DISTS 0.2686
  condition_l1 0.4113

k16 sparse control:
  bpp 0.014980
  control_bpp 0.000981
  PSNR 21.1959 / MS-SSIM 0.7116
  LPIPS 0.3664 / DISTS 0.2574
  condition_l1 0.4030

k16 sparse control + learned global condition gate:
  bpp 0.014978
  control_bpp 0.000978
  PSNR 21.2573 / MS-SSIM 0.7142
  LPIPS 0.3723 / DISTS 0.2622
  condition_l1 0.4037
  gate_mean 0.9085

k16 sparse control + learned spatial condition gate:
  bpp 0.014980
  control_bpp 0.000981
  PSNR 21.2206 / MS-SSIM 0.7127
  LPIPS 0.3679 / DISTS 0.2592
  condition_l1 0.4030
  gate_mean 0.9555

prefix16 + tail topk16 entropy control:
  bpp 0.015403
  control_bpp 0.001404
  PSNR 21.1966 / MS-SSIM 0.7117
  LPIPS 0.3659 / DISTS 0.2572
  condition_l1 0.4026
```

## Split Results

### k16 Sparse Control

```text
Kodak24:
  bpp 0.015086 / control_bpp 0.000963
  PSNR 20.8448 / MS-SSIM 0.6913
  LPIPS 0.3897 / DISTS 0.2601

CLIC2020 test428:
  bpp 0.014727 / control_bpp 0.000981
  PSNR 21.5709 / MS-SSIM 0.7243
  LPIPS 0.3573 / DISTS 0.2547

DIV2K val100:
  bpp 0.016036 / control_bpp 0.000985
  PSNR 19.6752 / MS-SSIM 0.6623
  LPIPS 0.3999 / DISTS 0.2683
```

### Spatial Gate

```text
Kodak24:
  bpp 0.015086 / control_bpp 0.000963
  PSNR 20.8705 / MS-SSIM 0.6928
  LPIPS 0.3910 / DISTS 0.2624

CLIC2020 test428:
  bpp 0.014727 / control_bpp 0.000981
  PSNR 21.5948 / MS-SSIM 0.7253
  LPIPS 0.3585 / DISTS 0.2564

DIV2K val100:
  bpp 0.016036 / control_bpp 0.000985
  PSNR 19.7030 / MS-SSIM 0.6636
  LPIPS 0.4024 / DISTS 0.2704
```

### Prefix16 + Tail TopK16

```text
Kodak24:
  bpp 0.015513 / control_bpp 0.001390
  PSNR 20.8473 / MS-SSIM 0.6913
  LPIPS 0.3891 / DISTS 0.2598

CLIC2020 test428:
  bpp 0.015153 / control_bpp 0.001406
  PSNR 21.5715 / MS-SSIM 0.7243
  LPIPS 0.3568 / DISTS 0.2545

DIV2K val100:
  bpp 0.016451 / control_bpp 0.001400
  PSNR 19.6759 / MS-SSIM 0.6624
  LPIPS 0.3993 / DISTS 0.2681
```

## Interpretation

`prefix_topk` is technically useful but not a breakthrough. It spends about
0.00042 more bpp than k16 and obtains only tiny LPIPS/DISTS gains. Compared
with k32, it reaches nearly the same perceptual point at lower bpp, so it is a
better high-control diagnostic than k32.

The spatial gate does not solve the main bottleneck. It improves PSNR/MS-SSIM
over raw k16 control and improves LPIPS/DISTS over the global gate, but it does
not dominate either one. Its learned gate mean is high, suggesting that the
model mostly chooses to pass the control residual and has not learned strong
content-specific suppression.

The main bottleneck is still condition/control quality, not the final gating
mechanism alone.

## Next Action

Move effort from small control packing and gate-only variants to a stronger
control-aware adapter path:

```text
1. train the adapter with simulated counted control in the loop
2. add condition residual statistics / spectral losses before image losses
3. use k16 sparse control as the efficient counted-control anchor
4. only revisit prefix_topk after the adapter learns to exploit the extra
   stable prefix information
5. build a proper multi-rate curve before any external-baseline claim
```

No external-baseline win should be claimed from these results.
