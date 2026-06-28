# 20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms2k_bs4_anchor025_rs005_tv005_chroma010_oi_only

Date: 2026-06-28T13:08:07

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage4_decoder_refiner.py --config configs/train/train_stage4_decoder_refiner.yaml --run-name 20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms2k_bs4_anchor025_rs005_tv005_chroma010_oi_only --crop-size 512 --batch-size 4 --max-steps 2000 --sample-every 500 --log-every 50 --wandb-mode offline
```

## Summary

- loss_total: 0.19267404079437256
- loss_l1: 0.05181197449564934
- loss_ms_ssim: 0.25137150287628174
- loss_lpips: 0.4131232500076294
- loss_dists: 0.0
- loss_anchor_stage3: 0.008228003047406673
- loss_refiner_tv: 0.01099930889904499
- loss_refiner_chroma: 0.0031219786033034325
- psnr_stage3: 21.358184814453125
- psnr_refined: 21.20694923400879
- psnr_delta_refined_vs_stage3: -0.15123558044433594
- l1_stage3: 0.050754647701978683
- l1_refined: 0.05181197449564934
- refiner_residual_abs_mean: 0.00823211669921875
- lr: 5e-05
- step: 2000.0
- grad_norm: 0.15387310087680817
- payload_policy: fixed decoder-side refiner weights; actual_payload_bpp unchanged from Stage 3

## Follow-Up Evaluation

- Kodak24 eval: `20260628_stage4_decoder_refiner_ms2k_bs4_midguard_oi_only_kodak24_eval`
- CLIC64 eval: `20260628_stage4_decoder_refiner_ms2k_bs4_midguard_oi_only_clic64_eval`
- Kodak24 Stage4 delta vs Stage3: PSNR -0.118058, MS-SSIM -0.007952, LPIPS -0.149847, DISTS -0.041942
- Kodak24 wins: LPIPS 24/24, DISTS 24/24, PSNR 0/24, MS-SSIM 0/24
- Kodak24 patch-FID: Stage3 215.036469, Stage4 165.095108
- CLIC64 Stage4 delta vs Stage3: PSNR -0.150638, MS-SSIM -0.007757, LPIPS -0.084864, DISTS -0.025630
- CLIC64 wins: LPIPS 57/64, DISTS 55/64, PSNR 0/64, MS-SSIM 1/64
- CLIC64 patch-FID: Stage3 174.378281, Stage4 186.570114

## Interpretation

This checkpoint is a perception-heavy diagnostic candidate. It improves LPIPS/DISTS strongly and Kodak patch-FID substantially at unchanged actual payload bpp, but it over-pushes the decoder refiner on CLIC64 and regresses CLIC patch-FID. Do not promote it as the default Stage 4 checkpoint without checkpoint selection or tighter validation gates.

## Artifacts

- checkpoint: `checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms2k_bs4_anchor025_rs005_tv005_chroma010_oi_only.pt`
- intermediate_checkpoints: `[]`
- output_dir: `outputs/stage4_decoder_refiner/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms2k_bs4_anchor025_rs005_tv005_chroma010_oi_only`
- summary: `outputs/stage4_decoder_refiner/20260628_stage4_decoder_refiner_cod512_gain050_lpips_ms2k_bs4_anchor025_rs005_tv005_chroma010_oi_only/summary.json`
