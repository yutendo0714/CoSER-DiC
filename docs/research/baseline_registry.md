# CoSER-DiC MVP-v0 Baseline Registry

Updated: 2026-06-27 JST

Canonical policy:

```text
docs/research/design_decisions/000_external_repos_and_pretrained_policy_mvp.md
```

Core rule:

```text
External pretrained models are tools, not the method.
```

Priority A for Core MVP:

- CompressAI Cheng2020 / hyperprior anchors
- CoD-Lite rate checkpoints and `CoD_Lite_pretrain.pt`
- CoD pixel/latent/one-step checkpoints as parallel heavy diffusion baseline,
  teacher, and upper-bound backbone candidates
- RDVQ checkpoints
- GLC pretrained
- StableCodec checkpoints

Priority B:

- AEIC
- Control-GIC
- OneDC
- RDEIC
- ResULIC

Default pretrained usage:

- Use `CoD_Lite_pretrain.pt` only as the diffusion decoder backbone
- initialization candidate for the default lightweight Stage-4 path.
- Run CoD in parallel as a heavier Stage-4 candidate / teacher / upper-bound
  track when resources permit. CoD is not allowed to replace the CoSER
  semantic/detail bitstream or bypass actual bpp accounting.
- Train CoSER semantic VQ, codebook, entropy models, detail residual branch,
  auxiliary decoders, and conditioning adapter from scratch.
- Do not initialize CoSER semantic VQ from RDVQ/GLC/Control-GIC in MVP-v0.
- Do not use StableCodec as the default decoder unless CoD-Lite integration
  fails.

Verification:

```bash
source .venv/bin/activate
python scripts/verify_mvp_assets.py
```

Download only the diffusion pretrain candidate:

```bash
source .venv/bin/activate
python scripts/download_cod_lite_assets.py --pretrain-only
```

Download all Core MVP CoD-Lite assets:

```bash
source .venv/bin/activate
python scripts/download_cod_lite_assets.py --all --include-yaml
```
