# CoD One-Step Official Full552 Baseline Plan

Date: 2026-06-29 JST

Status: planned; GPU execution is blocked until CUDA/NVML is visible again.

## Purpose

CoSER-DiC Stage 5 should be compared not only with CoD-Lite but also with the
official one-step CoD checkpoints when feasible. This plan mirrors the
CoD-Lite full552 baseline protocol:

- same CoSER 512x512 reference manifest
- same actual `.cod` payload bpp policy
- same split-specific patch-FID handling

## Generated Artifacts

Plan JSON:

```text
results/baselines/cod_one_step_official/20260629_cod512_full552_plan.json
```

Executable shell:

```text
results/baselines/cod_one_step_official/20260629_cod512_full552_plan.sh
```

Split manifests:

```text
results/baselines/cod_one_step_official/manifests/20260629_cod_one_step_official/full552.manifest.jsonl
results/baselines/cod_one_step_official/manifests/20260629_cod_one_step_official/kodak.manifest.jsonl
results/baselines/cod_one_step_official/manifests/20260629_cod_one_step_official/clic2020_test.manifest.jsonl
results/baselines/cod_one_step_official/manifests/20260629_cod_one_step_official/div2k_val.manifest.jsonl
```

Counts:

```text
full552: 552
kodak: 24
clic2020_test: 428
div2k_val: 100
```

## Checkpoints

Available deterministic one-step CoD checkpoints:

```text
0.00390625:
  external/pretrained/CoD/finetuned_one_step_cod/bpp_0_0039.pt

0.03125:
  external/pretrained/CoD/finetuned_one_step_cod/bpp_0_0312.pt

0.125:
  external/pretrained/CoD/finetuned_one_step_cod/bpp_0_1250.pt
```

The generated shell contains:

```text
3 bpp checkpoints x 4 protocol views = 12 eval runs
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

## GPU-Restart Command

Run after `nvidia-smi` and `torch.cuda.is_available()` are healthy:

```bash
bash results/baselines/cod_one_step_official/20260629_cod512_full552_plan.sh
```

The shared runner is:

```text
scripts/eval_cod_lite_official_baseline.py
```

For this plan it is invoked with:

```text
--official-repo external/repos/GenCodec/CoD
--official-module downstream.finetuned_one_step_cod
--codec-name "CoD one-step"
```

Payload policy:

```text
actual_payload_bpp =
  8 * (.cod_size - 4-byte width/height header) / pixels
```

Fixed model weights and fixed codebooks are excluded.
