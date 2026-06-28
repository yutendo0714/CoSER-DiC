# 012 Evaluation Protocol Alignment

Date: 2026-06-27 JST

## Decision

Separate CoSER-DiC evaluation into named protocols instead of using ambiguous
root-plus-limit conventions.

Active protocols:

- `coser_common_lic`
  - Kodak 24
  - CLIC Professional Validation 41
  - DIV2K validation 100
  - Metrics: PSNR, MS-SSIM, LPIPS, DISTS; optional FID/KID with explicit sample handling
  - bpp: `actual_payload_bpp`
- `coser_common_lic_plus_mobile`
  - Kodak 24
  - CLIC Professional Validation 41
  - CLIC Mobile Validation 61
  - DIV2K validation 100
  - Metrics and bpp same as `coser_common_lic`
- `cod_reproduction_512`
  - Kodak 24
  - CLIC2020 test 428
  - DIV2K validation 100
  - Preprocessing: resize if needed and center-crop every image to 512x512
  - Metrics: PSNR, LPIPS, DISTS, FID; optional MS-SSIM/KID
  - bpp: CoSER uses `actual_payload_bpp`; external codecs must label nominal,
    payload, and file bpp separately.
- `gencodec_reproduction`
  - Legacy alias for old commands; use `cod_reproduction_512` for new runs.
- CoD-Lite official/full-resolution comparison
  - Keep as a separate table, not a direct replacement for
    `cod_reproduction_512`.
  - Kodak uses Kodak512 / center-cropped setting.
  - CLIC2020 test is full-resolution.
  - DIV2K validation is appendix/additional.
  - Current crop-only CoSER evaluators should not claim this protocol until
    full-resolution padding/tiling evaluation is implemented.

Do not mix CLIC Professional Validation 41 with CLIC2020 test 428 in the same
table unless the table labels both explicitly.

## Why

The previous short-run CoSER evaluations used `/dpl/div2k --max-images 100`.
On this machine, `/dpl/div2k` is a flat mixed directory with `0001.png` through
`0900.png`. Taking the first 100 files therefore evaluates DIV2K train images,
not DIV2K validation.

The local CLIC layout is already sufficient for the CoD-style and
CoD-Lite-style CLIC2020 test
protocol:

```text
/dpl/clic/professional/test 250
/dpl/clic/mobile/test       178
total                       428
```

DIV2K validation is resolved from filename indices:

```text
/dpl/div2k/0801.png ... /dpl/div2k/0900.png
```

External references checked:

- TensorFlow Datasets CLIC catalog describes CLIC 2020 as a professional/mobile
  mixed learned image compression dataset:
  https://www.tensorflow.org/datasets/catalog/clic
- DIV2K official page defines 800 train, 100 validation, and 100 test images:
  https://data.vision.ee.ethz.ch/cvl/DIV2K/

## Implementation

Protocol resolver:

```text
src/coserdic/datasets/eval_protocols.py
```

Protocol config:

```text
configs/eval/protocols.yaml
configs/data/paths.yaml
```

Audit command:

```bash
source .venv/bin/activate
python scripts/dataset_protocol_report.py --strict
```

Main bitstream evaluation scripts now accept:

```bash
--eval-protocol cod_reproduction_512
--eval-dataset clic2020_test
--dpl-root /dpl
```

When `--eval-protocol` is used and `--max-images` is omitted, the full resolved
protocol is evaluated. Manual `--image-root` mode keeps the old small default
limit for quick smoke checks.

Updated scripts:

```text
scripts/eval_stage1_semantic_bitstream.py
scripts/eval_stage2_static_huffman_bitstream.py
scripts/eval_stage2_learned_topk_escape_bitstream.py
scripts/eval_stage3_uniform_residual_bitstream.py
scripts/eval_compressai_anchor.py
```

## Reporting Rule

For the primary CoSER generative-codec paper table, use the CoD paper-style
512 protocol:

```text
protocol: cod_reproduction_512
datasets: Kodak 24, CLIC2020 test 428, DIV2K validation 100
preprocessing: all images resize-if-needed + center-crop 512x512
metrics: PSNR, LPIPS, DISTS, FID
bpp: actual_payload_bpp for CoSER-DiC
```

FID handling for the CoD 512 table:

```text
FID is reported per dataset, not as one mixed Kodak+CLIC+DIV2K pool.

Patch sampling follows the CoD paper / released metric-code family:
  Kodak512:
    patch_size = 64
    fid_patch_num = 2
    samples = non-overlap 64 grid + 32px half-shift grid
  CLIC2020 test under CoD paper 512-crop protocol:
    patch_size = 128
    fid_patch_num = 2
    samples = non-overlap 128 grid + 64px half-shift grid
  DIV2K val:
    report only with explicit patch size/backend label unless the target
    baseline source defines the exact setting for that table.

This patch sampling applies to FID only.
PSNR, LPIPS, and DISTS stay image-level on the same 512x512 evaluation images.

CoSER's current metric script uses torch-fidelity, so values must be labeled
as CoD-style patch sampling with a torch-fidelity backend. For exact
official-code comparison, either add a torchmetrics FID backend or run the
external GenCodec/CoD metric script and label the backend.
```

For CoD-Lite official-code comparison, use a separate full-resolution table:

```text
protocol: cod_lite_official_fullres, currently documented but not implemented
as a crop-evaluator mode.

datasets:
  Kodak512 / center-cropped Kodak 24
  CLIC2020 test 428 full resolution
  DIV2K val 100 as appendix/additional

metrics:
  primary: LPIPS, DISTS, FID
  extended/appendix: PSNR, optional KID

FID:
  Kodak512: patch_size = 64, fid_patch_num = 2
  other datasets: patch_size = 256, fid_patch_num = 2

Do not mix this full-resolution CLIC table with the CoD 512-crop table.
```

For CoSER common LIC tables:

```text
datasets: Kodak 24, CLIC Professional Validation 41, DIV2K validation 100
metrics: PSNR, MS-SSIM, LPIPS, DISTS, optional FID/KID
bpp: actual_payload_bpp
```

Patch-based or overlapped-patch FID must be labeled in the table caption,
including dataset, patch size, shift/split count, and FID backend.
Kodak-only FID should be treated as a weak diagnostic because the sample count
is only 24.
