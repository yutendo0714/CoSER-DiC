# CoD-Lite Official Baseline, Kodak24 512

Date: 2026-06-28 JST  
Status: Baseline anchor for Core MVP Stage 4 / Stage 5

## Protocol

Input images:

```text
results/baselines/cod_lite_official/20260628_kodak24_stage3refs512_input
```

These are the same 24 Kodak center-cropped 512x512 reference images used by
the strict Stage 3 / Stage 4 CoSER evaluation manifest:

```text
results/bitstreams/stage3_uniform_residual/20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_gencodec512_no_pp_recon_export_prefixsafe/reconstructions/manifest.jsonl
```

Official codec path:

```text
external/repos/GenCodec/CoD_Lite/finetuned_one_step_codec/inference.py
compress -> .cod bitstream -> decompress
```

Bpp policy:

```text
actual_payload_bpp = 8 * (.cod_size - 4-byte width/height header) / pixels
codec_file_bpp    = 8 * .cod_size / pixels
```

Fixed model weights and fixed codebooks are excluded. This matches the CoSER
paper-bpp policy for entropy-coded payload streams.

Metrics:

```text
PSNR
MS-SSIM
LPIPS-Alex
DISTS
FID, GenCodec patch-FID protocol, Kodak patch_size=64, fid_patch_num=2
```

Runner:

```text
scripts/eval_cod_lite_official_baseline.py
```

## Results

| checkpoint | actual_payload_bpp | PSNR | MS-SSIM | LPIPS-Alex | DISTS | patch-FID |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CoD-Lite 0.0039 | 0.003906 | 19.0304 | 0.5700 | 0.3390 | 0.2089 | 51.2190 |
| CoD-Lite 0.0078 | 0.007812 | 19.7975 | 0.6283 | 0.2807 | 0.1733 | 44.6564 |
| CoD-Lite 0.0156 | 0.015625 | 20.7667 | 0.7090 | 0.2259 | 0.1402 | 38.4785 |
| CoD-Lite 0.0312 | 0.031250 | 21.9853 | 0.7811 | 0.1614 | 0.1120 | 31.8072 |

Artifacts:

```text
results/baselines/cod_lite_official/20260628_cod_lite_0039_kodak24_stage3refs512/summary.json
results/baselines/cod_lite_official/20260628_cod_lite_0078_kodak24_stage3refs512/summary.json
results/baselines/cod_lite_official/20260628_cod_lite_0156_kodak24_stage3refs512/summary.json
results/baselines/cod_lite_official/20260628_cod_lite_0312_kodak24_stage3refs512/summary.json
```

## CoSER Comparison Snapshot

Same Kodak24 reference images, 512x512:

| method | actual_payload_bpp | PSNR | MS-SSIM | LPIPS-Alex | DISTS | patch-FID |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CoSER Stage 3 no postprocess | 0.014121 | 21.6672 | 0.7225 | 0.6366 | 0.3732 | 216.7101 |
| CoSER Stage 4 lpips020 alpha0.1 | 0.014121 | 21.6785 | 0.7242 | 0.6204 | 0.3725 | 193.2827 |
| CoSER Stage 4 CoD-Lite adapter, lpips005 l1guard | 0.014121 | 20.5132 | 0.6822 | 0.4964 | 0.3151 | 117.5225 |
| CoD-Lite official 0.0156 | 0.015625 | 20.7667 | 0.7090 | 0.2259 | 0.1402 | 38.4785 |

Interpretation:

```text
CoSER Stage 4 has two distinct operating modes:
  alpha0.1 improves all Kodak24 means over Stage 3 without fidelity loss:
    FID 216.7 -> 193.3
    LPIPS 0.6366 -> 0.6204
    DISTS 0.3732 -> 0.3725
    PSNR 21.6672 -> 21.6785

  stronger raw/l1guard adapter improves perceptual metrics more:
    FID 216.7 -> 117.5
    LPIPS 0.6366 -> 0.4964
    DISTS 0.3732 -> 0.3151
  but loses fidelity and structure.

But it does not yet beat the official CoD-Lite 0.0156 checkpoint:
  CoD-Lite has much better LPIPS, DISTS, and patch-FID at similar bpp.
  CoSER keeps slightly lower payload bpp and better PSNR at alpha0.1, but
  remains far behind on perceptual quality.
```

Therefore the current Stage 4 result is useful evidence that the MVP
diffusion-backbone path is active, but it is not a Stage 5 / paper-claim
result.

## Next Baseline Work

1. Add the same official-baseline runner for CoD one-step checkpoints.
2. Add RDVQ official checkpoint evaluation on the same Kodak512 set.
3. Evaluate CoD-Lite on CLIC2020 test 428 and DIV2K val 100 once the exact
   input folders are finalized.
4. Keep CoSER bpp and official .cod payload/file bpp separately labeled.
