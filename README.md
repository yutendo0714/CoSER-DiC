# CoSER-DiC Research Workspace

This repository is the working implementation and experiment workspace for
**CoSER-DiC: Compression-Oriented Semantic-Residual Diffusion Codec**.

The original handoff package is preserved under
`docs/specs/coserdic_package_v1/`. The implementation lives under `src/coserdic`
so that external specifications, third-party baselines, checkpoints, and run
artifacts stay cleanly separated.

## Current Stage

Stage 0 for **Core MVP-v0**: environment, dataset inventory, reproducibility,
baseline registry, CoD-Lite pretrained policy, and actual-bitstream-first
evaluation scaffolding.

## Non-negotiable Research Axis

CoSER-DiC Core MVP-v0 must keep:

- entropy-constrained semantic VQ token stream
- detail residual latent stream
- semantic-conditioned residual entropy model
- auxiliary reconstruction as a fidelity anchor
- one-step or few-step compression-oriented diffusion reconstruction
- actual entropy-coded bitstream and actual bpp reporting

Core MVP-v0 deliberately excludes VLM-DPO, OCR/face losses, adaptive
quantization, advanced variable-rate control, full autoregressive/Mamba entropy
models, and large human studies until the MVP causal claims are established.

Default pretrained policy:

- `CoD_Lite_pretrain.pt` may initialize only the diffusion decoder backbone.
- Semantic VQ, codebook, entropy models, detail residual branch, auxiliary
  decoders, and CoSER conditioning adapters are scratch-trained.
- RDVQ, GLC, Control-GIC, StableCodec, AEIC, OneDC, CompressAI, ResULIC, and
  RDEIC are baselines/references unless a policy document explicitly says
  otherwise.

Forward-only estimated bpp is allowed for diagnostics, but paper-facing
evaluation must run:

```bash
x -> compress() -> bitstream -> decompress() -> metrics
```

## Layout

```text
configs/                 Experiment, dataset, and baseline configs
docs/specs/              Frozen CoSER-DiC handoff package
docs/research/           Literature survey, baseline notes, design decisions
docs/logs/               Work logs tied to runs/commands
src/coserdic/            Main Python package
scripts/                 Setup, dataset, baseline, and experiment utilities
tests/                   Unit tests for bitstream, padding, and infrastructure
external/repos/          Cloned third-party repositories, gitignored
external/pretrained/     Downloaded checkpoints/weights, gitignored
checkpoints/             CoSER-DiC checkpoints, gitignored
results/                 Metrics/tables/figures, gitignored
bitstreams/              Actual encoded streams, gitignored
artifacts/               Misc run artifacts, gitignored
wandb/                   Local W&B cache, gitignored
```

## Environment

Python is pinned to `3.10.12`, because it is the safest choice for PyTorch,
CompressAI, legacy LIC baselines, and perceptual metric packages.

```bash
pyenv local 3.10.12
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements/torch-cu124.txt
pip install -r requirements/base.txt
pip install -e .
python scripts/check_env.py
```

W&B is enabled by default in configs. Set the project explicitly if desired:

```bash
export WANDB_PROJECT=coserdic
export WANDB_MODE=online
```

For offline dry runs:

```bash
export WANDB_MODE=offline
```

## Data

Datasets are expected under `/dpl`. The initial configs point to the available
local datasets:

- `/dpl/kodak`
- `/dpl/clic/professional/valid`
- `/dpl/div2k`
- `/dpl/tecnick`
- `/dpl/coco30k`
- `/dpl/open-images-v6`
- `/dpl/textocr`

Run:

```bash
python scripts/dataset_report.py --config configs/data/paths.yaml
```

## Immediate Next Experiments

1. Validate environment and metric imports.
2. Verify/download CoD-Lite Core MVP assets.
3. Run CompressAI classical learned-codec anchors with actual bpp.
4. Implement Stage 1 semantic VQ tokenizer and evaluate semantic-only
   reconstructions.
5. Add actual semantic token bitstream round-trip before treating any rate point
   as valid.

Core MVP policy and baseline registry:

```bash
python scripts/verify_mvp_assets.py
python scripts/download_cod_lite_assets.py --dry-run --all --include-yaml
```
