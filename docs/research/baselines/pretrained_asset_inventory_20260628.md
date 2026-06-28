# Pretrained Asset Inventory

Date: 2026-06-28 JST

## Downloaded

GenCodec / CoD-Lite:

```text
external/pretrained/CoD_Lite/CoD_Lite_pretrain.pt
external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0039.pt
external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0039.yaml
external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0078.pt
external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0078.yaml
external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.pt
external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.yaml
external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0312.pt
external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0312.yaml
```

GenCodec / CoD:

```text
external/pretrained/CoD/cod/CoD_pixel_vpred.pt
external/pretrained/CoD/cod/CoD_pixel_vpred.yaml
external/pretrained/CoD/cod/CoD_latent_vpred.pt
external/pretrained/CoD/cod/CoD_latent_vpred.yaml
external/pretrained/CoD/cod/CoD_latent_vpred_64bits.pt
external/pretrained/CoD/cod/CoD_latent_vpred_64bits.yaml
external/pretrained/CoD/finetuned_one_step_cod/bpp_0_0039.pt
external/pretrained/CoD/finetuned_one_step_cod/bpp_0_0039.yaml
external/pretrained/CoD/finetuned_one_step_cod/bpp_0_0312.pt
external/pretrained/CoD/finetuned_one_step_cod/bpp_0_0312.yaml
external/pretrained/CoD/finetuned_one_step_cod/bpp_0_1250.pt
external/pretrained/CoD/finetuned_one_step_cod/bpp_0_1250.yaml
```

RDVQ:

```text
external/pretrained/RDVQ/lam_0.8_tau_0.01.pt
external/pretrained/RDVQ/lam_1.2_tau_0.01.pt
external/pretrained/RDVQ/lam_4.8_tau_0.1.pt
external/pretrained/RDVQ/lam_7.2_tau_0.1.pt
external/pretrained/RDVQ/lam_12_tau_0.1.pt
```

## Required Local Verification

Run:

```bash
.venv/bin/python scripts/verify_local_pretrained_assets.py --required-only --strict
```

This checks local files and expected sizes for the assets currently required by
the MVP Stage 4 / baseline plan. It is distinct from
`scripts/verify_mvp_assets.py`, which checks that remote assets exist.

For a full inventory check including optional CoD assets:

```bash
.venv/bin/python scripts/verify_local_pretrained_assets.py
```

Current verified local footprint:

```text
CoD:      37G
CoD-Lite: 2.3G
RDVQ:     4.7G
```

## Deferred

StableCodec:

```text
stablecodec_ft*.pkl
elic_official.pth
SD-Turbo
```

Reason: official checkpoints are on Google Drive and also require SD-Turbo.
Defer until the StableCodec baseline runner is implemented.

GLC:

```text
pretrained weights from GitHub release page
```

Reason: release assets need a targeted baseline runner and path mapping.

ResULIC / RDEIC:

```text
Stable Diffusion v2.1 + ModelScope/Google Drive checkpoints
```

Reason: heavier diffusion baselines with extra dependencies. Keep as secondary
baselines after Core MVP CoD/CoD-Lite and RDVQ anchors are operational.
