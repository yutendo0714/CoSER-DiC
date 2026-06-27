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
- `gencodec_reproduction`
  - Kodak 24
  - CLIC2020 test 428
  - DIV2K validation 100
  - Metrics: PSNR, LPIPS, DISTS, FID; optional MS-SSIM/KID
  - bpp: CoSER uses `actual_payload_bpp`; external codecs must label nominal,
    payload, and file bpp separately.

Do not mix CLIC Professional Validation 41 with CLIC2020 test 428 in the same
table unless the table labels both explicitly.

## Why

The previous short-run CoSER evaluations used `/dpl/div2k --max-images 100`.
On this machine, `/dpl/div2k` is a flat mixed directory with `0001.png` through
`0900.png`. Taking the first 100 files therefore evaluates DIV2K train images,
not DIV2K validation.

The local CLIC layout is already sufficient for the GenCodec/CoD-style test
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
--eval-protocol gencodec_reproduction
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

For GenCodec / CoD / CoD-Lite reproduction tables:

```text
datasets: Kodak 24, CLIC2020 test 428, DIV2K validation 100
metrics: PSNR, LPIPS, DISTS, FID
bpp: actual_payload_bpp for CoSER-DiC
```

For CoSER common LIC tables:

```text
datasets: Kodak 24, CLIC Professional Validation 41, DIV2K validation 100
metrics: PSNR, MS-SSIM, LPIPS, DISTS, optional FID/KID
bpp: actual_payload_bpp
```

Patch-based or overlapped-patch FID must be labeled in the table caption.
Kodak-only FID should be treated as a weak diagnostic because the sample count
is only 24.
