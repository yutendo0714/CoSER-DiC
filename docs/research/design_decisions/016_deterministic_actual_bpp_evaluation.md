# 016 Deterministic Actual-Bpp Evaluation

Date: 2026-06-27 JST

## Decision

Use deterministic evaluation by default for CoSER Stage 1/2/3 bitstream
evaluation scripts. Actual-payload-bpp tables should come from deterministic
runs unless explicitly labeled otherwise.

Added opt-out flag:

```bash
--allow-nondeterministic-eval
```

Affected scripts:

```text
scripts/eval_stage1_semantic_bitstream.py
scripts/eval_stage2_static_huffman_bitstream.py
scripts/eval_stage2_learned_topk_escape_bitstream.py
scripts/eval_stage3_uniform_residual_bitstream.py
```

`seed_everything(..., deterministic=True)` now sets deterministic torch
algorithms, disables cuDNN benchmark mode, enables deterministic cuDNN, and sets
`CUBLAS_WORKSPACE_CONFIG` when absent.

## Motivation

`actual_payload_bpp` is the main paper metric. It is computed from real
entropy-coded payload bytes, so even a one-token or one-residual-code difference
between repeated evaluations can change the reported bpp. That is unacceptable
for final tables unless the nondeterminism is explicitly audited.

## Audit Finding

Comparing an earlier `gain=1.1` perceptual run and a later reconstruction-export
run showed small but real mismatches:

```text
audit:
  results/analysis/reproducibility/20260627_gain110_perceptual_vs_recon_audit.json

rows: 165 / 165 matched by path
strict mismatches: 13
examples:
  detail_payload_bytes changed by 1 byte on:
    /dpl/clic/professional/valid/jared-erondu-21325.png
    /dpl/div2k/0842.png
  semantic_payload_bytes changed by 1 byte on:
    /dpl/div2k/0884.png
```

The mean bpp change was tiny, but it proved that nondeterministic eval can move
actual payload bytes.

## Deterministic Smoke

Two identical deterministic Stage 3 commands were run on the first 8 Kodak
images with `detail_gain=1.1`:

```text
20260627_stage3_deterministic_smoke_gain110_a
20260627_stage3_deterministic_smoke_gain110_b
```

Audit:

```text
results/analysis/reproducibility/20260627_stage3_deterministic_smoke_gain110_a_vs_b.json
```

Result:

```text
strict mismatches: 0
tolerance mismatches: 0
max absolute delta for all audited numeric fields: 0
pass: true
```

## Deterministic Active Anchor Refresh

The active Stage 3 b5 CoSER common LIC perceptual run was refreshed with
deterministic evaluation:

```text
run: 20260627_stage3_b5_semposleft_g4_coser_common_lic_deterministic_perceptual
wandb: wandb/offline-run-20260627_171536-85ydodw2
summary:
  results/bitstreams/stage3_uniform_residual/20260627_stage3_b5_semposleft_g4_coser_common_lic_deterministic_perceptual/summary.json
per-image:
  results/bitstreams/stage3_uniform_residual/20260627_stage3_b5_semposleft_g4_coser_common_lic_deterministic_perceptual/per_image_metrics.jsonl
```

Protocol:

```text
Kodak 24
CLIC Professional Validation 41
DIV2K validation 100
total: 165 images
crop: 256x256
```

Deterministic result:

```text
actual_payload_bpp: 0.018370
semantic_payload_bpp: 0.010518
detail_payload_bpp: 0.007852
debug_full_stream_bpp: 0.023375
PSNR: 21.390054
MS-SSIM: 0.693741
LPIPS Alex: 0.694714
DISTS: 0.409676
roundtrip failures: 0
```

The old nondeterministic run was nearly identical in means:

```text
old actual_payload_bpp: 0.018368
new actual_payload_bpp: 0.018370
old PSNR: 21.389952
new PSNR: 21.390054
old LPIPS: 0.694812
new LPIPS: 0.694714
```

But final paper tables should use the deterministic run.

## Reporting Rule

For CoSER-owned evaluation:

1. Use deterministic eval by default.
2. Use `actual_payload_bpp` from deterministic `compress/decompress`-style
   payload streams as `paper_bpp`.
3. Report nondeterministic or legacy runs only as historical/diagnostic.
4. If deterministic algorithms are disabled for speed, label the run and do not
   promote it to final paper tables without a reproducibility audit.
