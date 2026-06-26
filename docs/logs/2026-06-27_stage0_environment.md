# Stage 0 Environment / Foundation Log

Date: 2026-06-27 JST

W&B project: `coserdic`

Suggested run name after env install:

```text
20260627_coserdic_stage0_env_inventory_local
```

## Local Environment

- Workspace: `/workspace/CoSER-DiC`
- Python decision: `pyenv` Python `3.10.12`
- venv path: `.venv`
- Verified Python executable: `/workspace/CoSER-DiC/.venv/bin/python`
- PyTorch: `2.6.0+cu124`
- TorchVision: `0.21.0+cu124`
- CompressAI: `1.2.8`
- W&B: `0.28.0`
- NumPy: `1.26.4`
- GPU status at inspection: NVIDIA driver `550.144.03`, CUDA `12.4`,
  one RTX 4070 Ti SUPER with 16GB memory visible.
- `torch.cuda.is_available()`: `True`
- Existing global Python had no torch/compressai/wandb/metric packages.

Rationale:

- Python 3.10 is safer than 3.12 for CompressAI, torch entropy-coding
  extensions, old LIC baselines, and metric packages.
- PyTorch CUDA 12.4 wheels match the visible driver/CUDA stack.

## Project Structure

The original handoff package is copied to:

```text
docs/specs/coserdic_package_v1/
```

Implementation and experiments are separated into:

```text
src/coserdic/
configs/
scripts/
tests/
docs/research/
docs/logs/
external/repos/
external/pretrained/
checkpoints/
results/
bitstreams/
artifacts/
wandb/
```

## Dataset Inventory

Available under `/dpl`:

- Kodak: 24 images
- CLIC professional validation: 41 images
- CLIC professional test: 250 images
- DIV2K: 900 images
- Tecnick: 360 images
- COCO30k: 30,000 images
- FFHQ: 70,000 images
- OpenImages-v6: 351,620 images
- TextOCR root: 0 directly discoverable images in this path
- TextOCR-derived crop set: 64 images

No immediate additional dataset download is required for Stage 0-3.
For critical-detail dry runs, use `/dpl/race_task_textocr_crops` first.

## External Baseline Policy

External repositories are tracked in `configs/baselines/registry.yaml` and
should be cloned into `external/repos/` only when needed.

HEAD existence was verified for:

- CompressAI
- GLC
- RDVQ
- DLF
- ResULIC
- StableCodec
- GenCodec
- AEIC
- OneDC
- Control-GIC
- HPCM
- PSIC

## Implemented Foundation

- `CoSERBitstream`: outer stream container with JSON header, stream lengths,
  SHA-256 checksum, and actual bpp utility.
- Padding/cropping utilities for full-resolution evaluation.
- Recursive image-folder dataset wrapper.
- Dataset inventory script.
- Environment setup/check scripts.
- Baseline clone script.
- Initial survey and baseline notes.

## Next Verification

Completed:

```bash
source .venv/bin/activate
python scripts/check_env.py
python scripts/dataset_report.py --config configs/data/paths.yaml
pytest -q
```

Result:

```text
check_env.py: passed, CUDA visible
dataset_report.py: passed
pytest -q: 4 passed
```

CompressAI actual-bpp smoke test:

```bash
python scripts/eval_compressai_anchor.py \
  --dataset /dpl/kodak \
  --model bmshj2018_hyperprior \
  --quality 1 \
  --limit 1 \
  --output results/baselines/smoke_compressai_bmshj2018_hyperprior_q1_kodak1.json
```

Result:

```text
num_images: 1
mean_actual_bpp: 0.20377604166666666
mean_psnr_rgb: 25.34226551968043
mean_ms_ssim_rgb: 0.9159765243530273
```

If `torch.cuda.is_available()` is false despite `nvidia-smi` working outside the
container, pause experiments and ask for container restart before training.
