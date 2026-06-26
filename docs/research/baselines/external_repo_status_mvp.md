# External Repository Status for Core MVP-v0

Updated: 2026-06-27 JST

External repositories are cloned under `external/repos/` and are gitignored.
They are not part of the CoSER-DiC core implementation.

## Priority A Clone Status

| Key | Local path | HEAD | License | Core MVP role |
|---|---|---|---|---|
| cod_lite | `external/repos/GenCodec` | `c49eb0d643cc75e6c732cbc311a508627b54cf06` | MIT | diffusion backbone initialization reference and baseline |
| rdvq | `external/repos/RDVQ` | `a2744eedd803d9e20b55b4b0c935ec211a07d184` | MIT | VQ baseline and RD-aware VQ reference |
| glc | `external/repos/GLC` | `126db25f9c093508cd0c99cee32b53fd60074f9a` | Apache-2.0 | generative VQ baseline |
| stablecodec | `external/repos/StableCodec` | `73d9597cd97b8bca99c6c01a0156c37e6040e643` | MIT | one-step diffusion baseline and fallback reference |

CompressAI is installed from pip and not cloned by default.

## Dependency Caution

Do not install external repository requirements directly into the Core CoSER
`.venv`.

Observed version pressure:

- CoD-Lite requests `torch==2.5.0`, while Core CoSER uses `torch==2.6.0+cu124`.
- StableCodec requests `torch==2.1.2`, `torchvision==0.16.2`, older
  `diffusers`, and `xformers`.
- RDVQ and GLC include additional metric/baseline packages that should be
  isolated if needed.

Policy:

- Keep `.venv` as the CoSER-DiC core environment.
- Use external repos for reading, baseline commands, and checkpoint conversion.
- If a baseline requires incompatible dependencies, create a separate
  baseline-specific venv or container later.

## CoD-Lite Pretrain

Downloaded:

```text
external/pretrained/CoD_Lite/CoD_Lite_pretrain.pt
```

Size:

```text
786M
```

SHA-256:

```text
071c19eca6883b90c2ed2ffe5512c9a885c866bc2e723b55609ddb7768e46308
```

Usage:

- Allowed: initialize CoSER diffusion decoder backbone if architecture
  integration is feasible.
- Forbidden: treating CoD-Lite rate checkpoint outputs as CoSER-DiC final model
  results.

