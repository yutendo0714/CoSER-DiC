# Stage 4 No-Posthoc-RGB Mainline Policy

Date: 2026-06-28 JST  
Status: Active guardrail

## Decision

CoSER-DiC Stage 4 / Stage 5 should not treat post-hoc RGB output blending,
fixed alpha, output alpha prediction, or cosmetic decoder postprocess as the
proposal.

Those mechanisms may remain diagnostics, but the main method must live in the
codec and decoder-conditioning path:

```text
semantic/detail actual bitstream
  -> decoded CoSER feature state
  -> CoSER-owned condition adapter / control head
  -> pretrained CoD-Lite or CoD diffusion backbone
  -> reconstruction
```

The mainline question is:

```text
Can CoSER's transmitted semantic/detail information control a pretrained
compression diffusion decoder faithfully enough to improve the
rate-perception-faithfulness tradeoff?
```

It is not:

```text
Can we hide an unstable generative output by mixing it with Stage 3 RGB?
```

## Classification

Diagnostic only:

```text
fixed RGB alpha blend
learned RGB output alpha gate
decoder-side unsharp/gain postprocess as the Stage 4 method
visual-only postprocessing
```

Mainline:

```text
condition residual prediction
condition-space deterministic gate
semantic-latent / detail-context control
diffusion-friendly detail representation
adapter architecture and loss design
rate allocation that changes transmitted semantic/detail information
eventual counted control stream, if needed and if bits are counted
partial unfreeze / LoRA of CoD-Lite or CoD after adapter saturation
```

## Implementation Update

Added a condition-space gate:

```text
model:
  CoSERToCoDLiteConditionGate

script:
  scripts/train_stage4_cod_lite_condition_gate.py

eval:
  scripts/eval_stage4_cod_lite_adapter.py --condition-gate-checkpoint
```

The condition gate computes:

```text
ungated_pred_cond = base_cond + adapter_condition_residual
gate = gate(decoded CoSER tensors, base_cond, adapter_condition_residual)
gated_pred_cond = base_cond + gate * adapter_condition_residual
stage4 = CoD-Lite(stage3, gated_pred_cond)
```

There is no RGB output blend in this path.

Payload policy:

```text
condition gate weights are fixed decoder-side parameters
condition gate input is decoded from transmitted CoSER semantic/detail streams
no image-specific side information is transmitted
actual_payload_bpp remains the Stage 3 semantic/detail payload
```

## Probe Result

Short fidelity probe:

```text
train:
  20260628_stage4_cod_lite_condition_gate_fidelity_probe300_b1ga2

checkpoint:
  checkpoints/stage4_cod_lite_condition_gate/20260628_stage4_cod_lite_condition_gate_fidelity_probe300_b1ga2.pt

eval:
  20260628_stage4_cod_lite_condition_gate_fidelity_probe300_b1ga2_full552_eval

dataset:
  Kodak24 + CLIC2020 test428 + DIV2K val100

actual_payload_bpp:
  0.013999
```

Full552 result:

```text
Stage 3:
  PSNR / MS-SSIM: 21.9951 / 0.7348
  LPIPS / DISTS: 0.5758 / 0.3536

condition gate, no RGB blend:
  condition_gate_mean: 0.4327
  PSNR / MS-SSIM:     21.3169 / 0.7161
  LPIPS / DISTS:      0.5255 / 0.3437
```

Per-split trend:

```text
Kodak24:
  PSNR 21.6674 -> 21.0659
  MS-SSIM 0.7225 -> 0.7004
  LPIPS 0.6366 -> 0.5859
  DISTS 0.3732 -> 0.3661

CLIC2020 test428:
  PSNR 22.3638 -> 21.6774
  MS-SSIM 0.7454 -> 0.7283
  LPIPS 0.5610 -> 0.5090
  DISTS 0.3499 -> 0.3388

DIV2K val100:
  PSNR 20.4956 -> 19.8340
  MS-SSIM 0.6927 -> 0.6674
  LPIPS 0.6245 -> 0.5818
  DISTS 0.3648 -> 0.3594
```

Interpretation:

```text
the condition-space gate is a valid mainline mechanism
it improves perceptual metrics but is not faithful enough
it is not promoted as a Stage 4 win
it confirms that RGB blending was masking a real condition-control gap
```

## Detail-Aware Gate Follow-up

After the detail-aware adapter became the raw Stage 4 anchor, two gate probes
were run on top of:

```text
adapter:
  checkpoints/stage4_cod_lite_adapter/
  20260628_stage4_detailaware_adapter_idstart_ft600_b2.pt
```

Limit64 comparison:

| run | gate mean | PSNR | MS-SSIM | LPIPS | DISTS | condition_l1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ft600 raw | 0.0000 | 21.6722 | 0.7392 | 0.4142 | 0.2986 | 0.4170 |
| gate probe600 | 0.9645 | 21.6889 | 0.7397 | 0.4177 | 0.3011 | 0.4171 |
| gate mean035 probe400 | 0.5681 | 21.7718 | 0.7419 | 0.4684 | 0.3299 | 0.4367 |

Interpretation:

```text
probe600:
  collapses close to the raw adapter branch
  slightly improves fidelity but worsens LPIPS/DISTS

mean035 probe400:
  prevents full raw-branch collapse
  improves fidelity more
  substantially gives back LPIPS/DISTS and worsens condition_l1

current conclusion:
  gate mean regularization alone is not enough
  the next gate should be trained jointly with a better adapter objective or
  use a content-dependent fidelity/perception target rather than a fixed mean
```

Neither detail-aware gate probe is promoted.

## Next Mainline Work

Do not return to post-hoc RGB blend optimization.

Next work should target:

```text
1. stronger condition adapter trained end-to-end with decoded-image losses
2. condition-space gate trained jointly with the adapter, not only after it
3. diffusion-friendly detail control head from the same transmitted detail bits
4. testing CoD-Lite rate-specific checkpoints as decoder priors
5. exposing safe decoder conditioning hooks beyond a single condition tensor
6. larger non-eval train cache before drawing ceiling conclusions
```

Promotion requires:

```text
no RGB output blend
actual_payload_bpp unchanged unless extra bits are explicitly counted
full552 improvement over Stage 3
no hidden reference leakage
clear comparison to official CoD-Lite / CoD baselines
```
