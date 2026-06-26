# Core MVP-v0 Foundation Log

Date: 2026-06-27 JST

Source policy:

```text
docs/research/000_external_repos_and_pretrained_policy_mvp.md
docs/research/design_decisions/000_external_repos_and_pretrained_policy_mvp.md
```

## Decision

CoSER-DiC research is now narrowed to Core MVP-v0.

The method must remain a new CoSER-DiC codec, not a fork or wrapper around an
external codec.

Default pretrained use is restricted to:

```text
CoD_Lite_pretrain.pt -> diffusion decoder backbone initialization candidate
```

Scratch-trained CoSER-DiC components:

- semantic encoder/tokenizer
- VQ codebook
- semantic categorical entropy model
- detail residual encoder
- detail Gaussian entropy model
- semantic-conditioned residual prior
- semantic and joint auxiliary decoders
- CoSER conditioning adapter / FiLM / control branch
- actual bitstream coder

## Verified

- `zhaoyangjia/CoD_Lite` exists on Hugging Face.
- Expected CoD-Lite files are visible:
  - `CoD_Lite_pretrain.pt`
  - `CoD_Lite_bpp_0_0039.pt`
  - `CoD_Lite_bpp_0_0078.pt`
  - `CoD_Lite_bpp_0_0156.pt`
  - `CoD_Lite_bpp_0_0312.pt`
- Priority A repository HEADs were reachable:
  - GenCodec
  - RDVQ
  - GLC
  - StableCodec
  - RDEIC
- Environment remains healthy:
  - Python `3.10.12`
  - PyTorch `2.6.0+cu124`
  - CompressAI `1.2.8`
  - CUDA visible on RTX 4070 Ti SUPER
- Tests:
  - `pytest -q`: 10 passed
  - `python scripts/verify_mvp_assets.py`: passed
  - `python scripts/download_cod_lite_assets.py --dry-run --all --include-yaml`: passed
- Cloned Priority A external repositories:
  - `external/repos/GenCodec`
  - `external/repos/RDVQ`
  - `external/repos/GLC`
  - `external/repos/StableCodec`
- Downloaded Core MVP default pretrained candidate:
  - `external/pretrained/CoD_Lite/CoD_Lite_pretrain.pt`
  - size: `786M`
  - sha256: `071c19eca6883b90c2ed2ffe5512c9a885c866bc2e723b55609ddb7768e46308`
- Dataset report:
  - Kodak: 24
  - CLIC valid/test: 41 / 250
  - DIV2K: 900
  - Tecnick: 360
  - COCO30k: 30,000
  - FFHQ: 70,000
  - OpenImages-v6: 351,620
  - TextOCR-derived crops: 64

## Added

- Core MVP baseline registry in `configs/baselines/registry.yaml`
- Baseline config stubs for CompressAI, CoD-Lite, RDVQ, GLC, StableCodec
- Pretrained path registry in `configs/paths/pretrained.yaml`
- CoD-Lite asset downloader and verifier scripts
- Core MVP design audit
- CoSER-owned conditioning adapter and diffusion decoder boundary
- Extra sanity tests for side-info leakage, deterministic bitstream unpack,
  estimated/actual bpp gap, and adapter shape

## Cleanup

To keep the active tree focused on Core MVP-v0:

- Removed duplicate root policy file after confirming it was byte-identical to
  `docs/research/design_decisions/000_external_repos_and_pretrained_policy_mvp.md`.
- Removed the older `baseline_registry_notes.md`, which conflicted with the
  Core MVP priority ordering.
- Moved the broad LIC survey to `docs/research/archive/`.
- Moved the Stage 0 design memo to `docs/research/design_decisions/`.
- Moved variable-rate training config to `configs/train/post_mvp/`.
- Added small README files for `docs/research/`, `docs/research/archive/`,
  `configs/train/`, and `configs/baselines/`.

## Next Commands

Verify assets without downloading:

```bash
source .venv/bin/activate
python scripts/verify_mvp_assets.py
python scripts/download_cod_lite_assets.py --dry-run --all --include-yaml
```

Download the only default initialization candidate:

```bash
source .venv/bin/activate
python scripts/download_cod_lite_assets.py --pretrain-only
```

Clone only Priority A external repos when baseline reproduction begins:

```bash
source .venv/bin/activate
python scripts/clone_baselines.py --priority A
```
