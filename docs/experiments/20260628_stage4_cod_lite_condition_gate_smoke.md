# 20260628_stage4_cod_lite_condition_gate_smoke

Date: 2026-06-28T23:00:56

## Command

```bash
scripts/train_stage4_cod_lite_condition_gate.py --adapter-checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_statsmatch_ft2k_b8.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_traincache_clic_oi_2048_seed20260628/reconstructions/manifest.jsonl --run-name 20260628_stage4_cod_lite_condition_gate_smoke --crop-size 512 --limit 4 --batch-size 1 --grad-accum-steps 2 --num-workers 0 --max-steps 2 --lr 1e-4 --hidden-channels 32 --num-blocks 1 --init-gate 0.3 --image-l1-weight 1.0 --lpips-weight 0.0 --ms-ssim-weight 0.0 --condition-l1-weight 0.1 --stage3-l1-guard-weight 1.0 --stage3-mse-guard-weight 2.0 --wandb-mode offline
```

## Summary

- loss_mean: 0.11424986273050308
- image_l1_mean: 0.059330904856324196
- lpips_mean: 0.0
- ms_ssim_loss_mean: 0.2728393226861954
- condition_l1_mean: 0.4460045099258423
- ungated_condition_l1_mean: 0.37949250638484955
- condition_regression_guard_mean: 0.06651200354099274
- stage3_l1_guard_mean: 0.007387865334749222
- stage3_mse_guard_mean: 0.0014653195394203067
- gate_mean_loss_mean: 0.0
- stage4_psnr_mean: 21.4132399559021
- stage3_psnr_mean: 22.366272926330566
- stage4_ms_ssim_mean: 0.7271606773138046
- stage3_ms_ssim_mean: 0.7537274658679962
- gate_mean_mean: 0.30078125
- gate_std_mean: 0.0
- gate_min_mean: 0.30078125
- gate_max_mean: 0.30078125
- gate_mean_over_steps: 0.30078125
- gate_mean_std_over_steps: 0.0
- payload_policy: deterministic decoder-side condition gate from decoded CoSER tensors and fixed model weights; no transmitted side information and no RGB output blending

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_condition_gate/20260628_stage4_cod_lite_condition_gate_smoke.pt`
- summary: `results/stage4_cod_lite_condition_gate/20260628_stage4_cod_lite_condition_gate_smoke/summary.json`
- results_dir: `results/stage4_cod_lite_condition_gate/20260628_stage4_cod_lite_condition_gate_smoke`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_230054-bpdlaxn4`
