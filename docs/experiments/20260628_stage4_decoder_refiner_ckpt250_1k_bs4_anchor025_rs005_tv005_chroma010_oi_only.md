# 20260628_stage4_decoder_refiner_ckpt250_1k_bs4_anchor025_rs005_tv005_chroma010_oi_only

Date: 2026-06-28T13:26:59

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage4_decoder_refiner.py --config configs/train/train_stage4_decoder_refiner.yaml --run-name 20260628_stage4_decoder_refiner_ckpt250_1k_bs4_anchor025_rs005_tv005_chroma010_oi_only --crop-size 512 --batch-size 4 --max-steps 1000 --checkpoint-every 250 --sample-every 250 --log-every 50 --wandb-mode offline
```

## Summary

- loss_total: 0.20854312181472778
- loss_l1: 0.05055537447333336
- loss_ms_ssim: 0.2684427499771118
- loss_lpips: 0.45696592330932617
- loss_dists: 0.0
- loss_anchor_stage3: 0.006794673856347799
- loss_refiner_tv: 0.00917118787765503
- loss_refiner_chroma: 0.004409430548548698
- psnr_stage3: 21.822912216186523
- psnr_refined: 21.717233657836914
- psnr_delta_refined_vs_stage3: -0.10567855834960938
- l1_stage3: 0.04976768419146538
- l1_refined: 0.05055537447333336
- refiner_residual_abs_mean: 0.00679779052734375
- lr: 5e-05
- step: 1000.0
- grad_norm: 0.1530524641275406
- payload_policy: fixed decoder-side refiner weights; actual_payload_bpp unchanged from Stage 3

## Checkpoint Selection

CLIC64 actual-bitstream screen:

- step250: LPIPS -0.051800, DISTS -0.018209, PSNR -0.073260, MS-SSIM -0.002857, FID 180.647430
- step500: LPIPS -0.056711, DISTS -0.017118, PSNR -0.071926, MS-SSIM -0.002724, FID 172.581406
- step750: LPIPS -0.067543, DISTS -0.020312, PSNR -0.090251, MS-SSIM -0.004066, FID 175.158325
- step1000: LPIPS -0.075625, DISTS -0.025473, PSNR -0.115118, MS-SSIM -0.004917, FID 178.079422

Kodak24 confirmation for step500:

- LPIPS -0.092378, DISTS -0.021814, PSNR -0.049467, MS-SSIM -0.002532
- FID: Stage3 215.036469, Stage4 177.938614

Decision: use `step000500.pt` as the current balanced Stage 4 checkpoint candidate. It improves CLIC64 patch-FID against Stage 3 and gives a stronger Kodak FID gain than the 300-step mid-guard checkpoint without drifting as far as the 2k perception-heavy checkpoint.

## Artifacts

- checkpoint: `checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_ckpt250_1k_bs4_anchor025_rs005_tv005_chroma010_oi_only.pt`
- intermediate_checkpoints: `['checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_ckpt250_1k_bs4_anchor025_rs005_tv005_chroma010_oi_only_step000250.pt', 'checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_ckpt250_1k_bs4_anchor025_rs005_tv005_chroma010_oi_only_step000500.pt', 'checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_ckpt250_1k_bs4_anchor025_rs005_tv005_chroma010_oi_only_step000750.pt']`
- output_dir: `outputs/stage4_decoder_refiner/20260628_stage4_decoder_refiner_ckpt250_1k_bs4_anchor025_rs005_tv005_chroma010_oi_only`
- summary: `outputs/stage4_decoder_refiner/20260628_stage4_decoder_refiner_ckpt250_1k_bs4_anchor025_rs005_tv005_chroma010_oi_only/summary.json`
