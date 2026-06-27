# 013 Stage 3 Protocol Perceptual Evaluation

Date: 2026-06-27 JST

## Decision

Keep two Stage 3 anchors:

- Quality anchor: `d32/b5/r0.25 semantic_position_leftctx_huffman g4 smoothing=0`
- Low-rate anchor: `d32/b4/r0.25 semantic_position_leftctx_huffman g4 smoothing=0`

Do not claim SOTA superiority yet. These anchors establish a correct actual
bitstream, protocol-aligned, perceptual-metric baseline for the Core MVP before
moving to Stage 4 generative decoding.

## Implementation Update

Added optional LPIPS/DISTS evaluation:

```text
src/coserdic/metrics/perceptual.py
scripts/eval_stage1_semantic_bitstream.py
scripts/eval_stage2_static_huffman_bitstream.py
scripts/eval_stage2_learned_topk_escape_bitstream.py
scripts/eval_stage3_uniform_residual_bitstream.py
scripts/eval_compressai_anchor.py
```

Use:

```bash
--compute-perceptual
```

The scripts only report LPIPS/DISTS when the flag is present. This avoids
accidental zero-valued perceptual metrics in fast bpp/roundtrip smoke runs.

Analysis utilities:

```text
scripts/summarize_per_image_metrics.py
scripts/compare_stage3_per_image_metrics.py
```

`compare_stage3_per_image_metrics.py` now supports:

```bash
--quality-direction higher
--quality-direction lower
```

for PSNR/MS-SSIM versus LPIPS/DISTS.

## CoSER Common LIC Protocol Results

Protocol:

```text
Kodak 24
CLIC Professional Validation 41
DIV2K validation 100, filenames 0801-0900
total: 165 images
crop: 256x256 center crop
bpp: actual_payload_bpp
```

Quality anchor b5:

```text
run: 20260627_stage3_b5_semposleft_g4_coser_common_lic_perceptual
wandb: wandb/offline-run-20260627_163024-mpf68qex
summary:
  results/bitstreams/stage3_uniform_residual/20260627_stage3_b5_semposleft_g4_coser_common_lic_perceptual/summary.json
per-image:
  results/bitstreams/stage3_uniform_residual/20260627_stage3_b5_semposleft_g4_coser_common_lic_perceptual/per_image_metrics.jsonl

actual_payload_bpp: 0.018368
semantic_payload_bpp: 0.010516
detail_payload_bpp: 0.007852
debug_full_stream_bpp: 0.023373
PSNR: 21.389952
PSNR delta vs semantic-only: +0.471572 dB
MS-SSIM: 0.693731
MS-SSIM delta: +0.004616
LPIPS Alex: 0.694812
LPIPS delta: -0.005412
DISTS: 0.409673
DISTS delta: -0.005049
roundtrip failures: 0
```

Low-rate anchor b4:

```text
run: 20260627_stage3_b4_semposleft_g4_coser_common_lic_perceptual
wandb: wandb/offline-run-20260627_163142-itygc56s
summary:
  results/bitstreams/stage3_uniform_residual/20260627_stage3_b4_semposleft_g4_coser_common_lic_perceptual/summary.json
per-image:
  results/bitstreams/stage3_uniform_residual/20260627_stage3_b4_semposleft_g4_coser_common_lic_perceptual/per_image_metrics.jsonl

actual_payload_bpp: 0.015678
semantic_payload_bpp: 0.010516
detail_payload_bpp: 0.005162
debug_full_stream_bpp: 0.020683
PSNR: 21.313472
PSNR delta vs semantic-only: +0.395093 dB
MS-SSIM: 0.692550
MS-SSIM delta: +0.003435
LPIPS Alex: 0.699392
LPIPS delta: -0.000832
DISTS: 0.409501
DISTS delta: -0.005221
roundtrip failures: 0
```

b5 versus b4:

```text
bpp cost: +0.002689 actual_payload_bpp
PSNR: +0.076480 dB, b5 wins 165/165 images
LPIPS: -0.004580, b5 wins 146/165 images
DISTS: +0.000172, b5 wins 78/165 images
```

Interpretation:

- b5 is the safer quality/perceptual anchor.
- b4 is a useful low-rate anchor and has essentially tied or slightly better
  DISTS on this protocol.
- b4 shows more LPIPS regressions, especially on DIV2K validation, so it should
  not be the main perceptual-quality claim.

Stage 3 versus semantic-only:

```text
b5:
  PSNR improves on 164/165 images
  MS-SSIM improves on 162/165 images
  LPIPS improves on 127/165 images
  DISTS improves on 144/165 images

b4:
  PSNR improves on 164/165 images
  MS-SSIM improves on 153/165 images
  LPIPS improves on 86/165 images
  DISTS improves on 142/165 images
```

## GenCodec Reproduction Protocol Result

Protocol:

```text
Kodak 24
CLIC2020 test 428 = professional/test 250 + mobile/test 178
DIV2K validation 100, filenames 0801-0900
total: 552 images
crop: 256x256 center crop for current CoSER Core MVP evaluation
bpp: actual_payload_bpp
```

b5 quality anchor:

```text
run: 20260627_stage3_b5_semposleft_g4_gencodec_reproduction_perceptual
wandb: wandb/offline-run-20260627_163727-0d979w5s
summary:
  results/bitstreams/stage3_uniform_residual/20260627_stage3_b5_semposleft_g4_gencodec_reproduction_perceptual/summary.json
per-image:
  results/bitstreams/stage3_uniform_residual/20260627_stage3_b5_semposleft_g4_gencodec_reproduction_perceptual/per_image_metrics.jsonl

actual_payload_bpp: 0.017736
semantic_payload_bpp: 0.009954
detail_payload_bpp: 0.007782
debug_full_stream_bpp: 0.022742
PSNR: 22.137718
PSNR delta vs semantic-only: +0.554524 dB
MS-SSIM: 0.722216
MS-SSIM delta: +0.005370
LPIPS Alex: 0.647944
LPIPS delta: -0.006018
DISTS: 0.401050
DISTS delta: -0.004718
roundtrip failures: 0
```

Dataset breakdown:

```text
Kodak 24:
  actual_payload_bpp: 0.018473
  LPIPS delta: -0.008245
  DISTS delta: -0.005355

CLIC2020 professional test 250:
  actual_payload_bpp: 0.017369
  LPIPS delta: -0.007098
  DISTS delta: -0.004708

CLIC2020 mobile test 178:
  actual_payload_bpp: 0.017555
  LPIPS delta: -0.005164
  DISTS delta: -0.004469

DIV2K validation 100:
  actual_payload_bpp: 0.018798
  LPIPS delta: -0.004302
  DISTS delta: -0.005035
```

## CompressAI Anchor

Reference run:

```text
model: bmshj2018_hyperprior
quality: 1
dataset: Kodak 24
output:
  results/baselines/compressai_bmshj2018_hyperprior_q1_kodak24_perceptual.json

actual_payload_bpp: 0.131300
PSNR: 27.530324
MS-SSIM: 0.916931
LPIPS Alex: 0.380007
DISTS: 0.230248
```

This is not a matched-rate comparison. It is an RD anchor showing that the
current CoSER Stage 3 result is an ultra-low-rate stream, not yet a perceptual
SOTA codec.

## Research Interpretation

The current Core MVP has a solid and auditable Stage 1-3 bitstream:

- actual payload bpp is below 0.02 on all checked protocols
- semantic and detail payloads roundtrip exactly
- CLIC/DIV2K/Kodak protocol handling is now explicit
- Stage 3 residuals improve PSNR/MS-SSIM/DISTS on most images

However, current perceptual quality is far from strong RD codecs at much higher
rates, and probably far from generative codecs at similar rates. The present
residual grid is a clean fidelity anchor, not the final perceptual mechanism.

Before Stage 4:

1. Keep b5 as the quality anchor and b4 as the low-rate anchor.
2. Inspect LPIPS regressions, especially DIV2K images where b4 worsens LPIPS.
3. Add reconstruction export plus FID/KID or patch-FID only after sample
   handling is specified.
4. Do not optimize Stage 3 into a full pixel codec; preserve semantic/detail
   stream separation for the Stage 4 generative decoder.
