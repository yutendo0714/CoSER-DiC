# CoD-Lite Official Full552 Baseline Plan

Date: 2026-06-29 JST

Status: planned; GPU execution is blocked until CUDA/NVML is visible again.

## Purpose

Stage 5 claims need a direct official CoD-Lite curve on the same CoD-style
512x512 evaluation images used by CoSER:

- Kodak 24
- CLIC2020 test 428
- DIV2K validation 100
- total 552 images

This plan keeps official CoD-Lite `.cod` payload bpp separate from CoSER
semantic/detail payload bpp, but both are reported as actual payload bpp:

```text
actual_payload_bpp =
  8 * (.cod_size - 4-byte width/height header) / pixels
```

Fixed model weights and fixed codebooks are excluded.

## Generated Artifacts

Planner:

```text
scripts/plan_cod_lite_official_baselines.py
```

Plan JSON:

```text
results/baselines/cod_lite_official/20260629_cod512_full552_plan.json
```

Executable shell:

```text
results/baselines/cod_lite_official/20260629_cod512_full552_plan.sh
```

Split manifests:

```text
results/baselines/cod_lite_official/manifests/20260629_cod_lite_official/full552.manifest.jsonl
results/baselines/cod_lite_official/manifests/20260629_cod_lite_official/kodak.manifest.jsonl
results/baselines/cod_lite_official/manifests/20260629_cod_lite_official/clic2020_test.manifest.jsonl
results/baselines/cod_lite_official/manifests/20260629_cod_lite_official/div2k_val.manifest.jsonl
```

Counts:

```text
full552: 552
kodak: 24
clic2020_test: 428
div2k_val: 100
```

## Evaluation Layout

The generated shell contains 16 official CoD-Lite evaluations:

```text
4 bpp checkpoints x 4 protocol views
```

Bpp checkpoints:

```text
0.00390625
0.0078125
0.015625
0.03125
```

Protocol views:

```text
full552 aggregate:
  fid_patch_size = 0
  use for PSNR / MS-SSIM / LPIPS / DISTS curve

kodak:
  fid_patch_size = 64
  fid_patch_num = 2

clic2020_test:
  fid_patch_size = 256
  fid_patch_num = 2

div2k_val:
  fid_patch_size = 256
  fid_patch_num = 2
```

Rationale:

```text
Do not mix Kodak patch64 and CLIC/DIV2K patch256 into one aggregate FID.
Patch extraction is used for FID only. PSNR, MS-SSIM, LPIPS, and DISTS use the
normal image-level evaluation.
```

## GPU-Restart Command

Run after `nvidia-smi` and `torch.cuda.is_available()` are healthy:

```bash
bash results/baselines/cod_lite_official/20260629_cod512_full552_plan.sh
```

The shell also collects curve JSON files after the eval summaries exist:

```text
results/baselines/cod_lite_official/curves/20260629_cod_lite_official_cod_lite_cod512_full552_curve.json
results/baselines/cod_lite_official/curves/20260629_cod_lite_official_cod_lite_cod512_kodak_curve.json
results/baselines/cod_lite_official/curves/20260629_cod_lite_official_cod_lite_cod512_clic2020_test_curve.json
results/baselines/cod_lite_official/curves/20260629_cod_lite_official_cod_lite_cod512_div2k_val_curve.json
```

## Current Kodak Anchor

The existing Kodak512 official curve remains:

```text
docs/research/baselines/cod_lite_kodak512_official_curve.json
```

It was updated to state the actual `.cod` payload bpp policy and include
PSNR/MS-SSIM for all stored points.
