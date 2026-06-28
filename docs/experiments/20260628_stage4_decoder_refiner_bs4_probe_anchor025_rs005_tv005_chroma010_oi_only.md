# 20260628_stage4_decoder_refiner_bs4_probe_anchor025_rs005_tv005_chroma010_oi_only

Date: 2026-06-28T12:55:41

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage4_decoder_refiner.py --config configs/train/train_stage4_decoder_refiner.yaml --run-name 20260628_stage4_decoder_refiner_bs4_probe_anchor025_rs005_tv005_chroma010_oi_only --crop-size 512 --batch-size 4 --max-steps 10 --residual-scale 0.05 --loss-anchor-stage3 0.25 --loss-refiner-tv 0.05 --loss-refiner-chroma 0.10 --sample-every 10 --log-every 5 --wandb-mode offline
```

## Summary

- loss_total: 0.2055482715368271
- loss_l1: 0.04012800753116608
- loss_ms_ssim: 0.2055755853652954
- loss_lpips: 0.5079302191734314
- loss_dists: 0.0
- loss_anchor_stage3: 4.109650762984529e-05
- loss_refiner_tv: 4.721764980786247e-06
- loss_refiner_chroma: 5.1786140829790384e-05
- psnr_stage3: 22.538013458251953
- psnr_refined: 22.537979125976562
- psnr_delta_refined_vs_stage3: -3.4332275390625e-05
- l1_stage3: 0.040130339562892914
- l1_refined: 0.04012800753116608
- refiner_residual_abs_mean: 4.1604042053222656e-05
- lr: 5e-05
- step: 10.0
- grad_norm: 0.011249965988099575
- payload_policy: fixed decoder-side refiner weights; actual_payload_bpp unchanged from Stage 3

## Artifacts

- checkpoint: `checkpoints/stage4_decoder_refiner/20260628_stage4_decoder_refiner_bs4_probe_anchor025_rs005_tv005_chroma010_oi_only.pt`
- output_dir: `outputs/stage4_decoder_refiner/20260628_stage4_decoder_refiner_bs4_probe_anchor025_rs005_tv005_chroma010_oi_only`
- summary: `outputs/stage4_decoder_refiner/20260628_stage4_decoder_refiner_bs4_probe_anchor025_rs005_tv005_chroma010_oi_only/summary.json`
