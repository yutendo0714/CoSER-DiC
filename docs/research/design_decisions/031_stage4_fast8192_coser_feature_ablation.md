# Stage 4 Fast8192 CoSER Feature-Use Ablation

Date: 2026-06-29

## Decision

The fast8192 Stage 4 adapter is using the decoded CoSER semantic latent in a
meaningful image-specific way. The decoded detail context is useful but still
under-used.

Next mainline work should strengthen diffusion-friendly detail conditioning
rather than spend time on RGB blending or generic perceptual loss tuning.

## Checkpoint

```text
checkpoints/stage4_cod_lite_adapter/
20260629_stage4_detailaware_ft600_fast8192_ft1500_b4.pt
```

Evaluation split:

```text
Kodak24 + CLIC2020 test428 + DIV2K val100
crop: 512x512
actual_payload_bpp / paper_bpp: 0.013999
```

## Ablations

All runs use the same Stage 3 payload and the same frozen CoD-Lite backbone.
Only decoder-side adapter inputs are ablated.

| ablation | PSNR | MS-SSIM | LPIPS | DISTS | condition_l1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| normal | 21.3957 | 0.7186 | 0.4342 | 0.3022 | 0.4095 |
| semantic latent zero | 20.3170 | 0.6961 | 0.4624 | 0.3396 | 0.4269 |
| semantic latent shuffle | 20.1303 | 0.6990 | 0.4607 | 0.3173 | 0.4339 |
| detail context zero | 21.4080 | 0.7182 | 0.4394 | 0.3061 | 0.4105 |
| detail context shuffle | 21.3836 | 0.7179 | 0.4377 | 0.3045 | 0.4107 |

## Interpretation

Semantic latent:

```text
zero / shuffle cause large fidelity and perceptual regressions
condition_l1 worsens by about 0.017 to 0.024
the adapter is not merely relying on x_aux or base_cond
image-specific semantic correspondence matters
```

Detail context:

```text
zero / shuffle cause small but consistent LPIPS/DISTS regressions
condition_l1 worsens only slightly
PSNR can stay flat or rise because detail context mainly helps perceptual
condition recovery rather than conservative fidelity
```

This confirms that the CoSER semantic stream is on the mainline, while the
detail stream is not yet strong enough as a diffusion-control signal.

## Consequence

Promote the following diagnosis:

```text
semantic VQ stream:
  already provides important CoD-Lite condition signal

detail residual stream:
  valid but weak for Stage 4
  needs a diffusion-control head / stronger detail-aware training
```

Do not interpret this as a reason to remove detail bits. The detail stream is
needed for Stage 3 fidelity and should become more useful for Stage 4.

## Next Experiments

Short probes:

```text
1. train a continuation with semantic-latent dropout / corruption
   so the adapter must exploit detail context more

2. add an auxiliary detail-control projection head from the same transmitted
   detail codes, without adding payload bits

3. require promotion to improve normal full552 metrics and increase detail
   ablation sensitivity without damaging semantic ablation sensitivity
```

Medium path:

```text
unfreeze detail decoder / projection before changing entropy-coded detail
tokens
keep actual_payload_bpp unchanged until a counted control stream is introduced
```

This keeps the MVP information flow intact:

```text
CoSER semantic VQ stream + CoSER detail residual stream
-> CoSER-owned condition adapter
-> pretrained CoD-Lite / CoD diffusion backbone
```
