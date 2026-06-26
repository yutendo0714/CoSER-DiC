# Baseline Registry Notes

Updated: 2026-06-27 JST

The registry in `configs/baselines/registry.yaml` separates third-party
baselines from the CoSER-DiC implementation. External repositories should be
cloned into `external/repos/` and kept gitignored.

Priority A for the MVP:

- `compressai`: RD-oriented LIC anchors and classical codec wrappers.
- `glc`: VQ-VAE latent-space generative coding.
- `rdvq`: differentiable VQ with rate feedback.
- `dlf`: semantic/detail dual generative latent fusion.
- `resulic`: semantic residual coding plus compression-aware diffusion.
- `stablecodec`: one-step diffusion codec.
- `gencodec`: CoD/CoD-Lite diffusion foundation and real-time generative codec.
- `aeic`: shallow/moderate encoder ultra-low bitrate diffusion codec.

Rules:

- Do not merge external code into `src/coserdic`.
- Record commit hash, license, checkpoint URL, and exact inference command
  before using a baseline in a main table.
- Prefer official compress/decompress or entropy-coded inference paths.
- Mark forward-only or estimated-bpp results as diagnostic only.

Clone Priority A candidates:

```bash
python scripts/clone_baselines.py --priority A
```

Clone a single candidate:

```bash
python scripts/clone_baselines.py --name rdvq
```

