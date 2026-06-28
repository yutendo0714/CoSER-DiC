# 20260628_stage4_cod_lite_gate_statsmatch_fidelity_600_b1ga2

Date: 2026-06-28T22:26:52

## Command

```bash
scripts/train_stage4_cod_lite_gate.py --adapter-checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_statsmatch_ft2k_b8.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_traincache_clic_oi_2048_seed20260628/reconstructions/manifest.jsonl --run-name 20260628_stage4_cod_lite_gate_statsmatch_fidelity_600_b1ga2 --crop-size 512 --batch-size 1 --grad-accum-steps 2 --num-workers 2 --max-steps 600 --lr 1e-4 --hidden-channels 64 --num-blocks 1 --alpha-min 0.0 --alpha-max 0.45 --init-alpha 0.3 --image-l1-weight 1.0 --lpips-weight 0.04 --ms-ssim-weight 0.3 --stage3-l1-guard-weight 6.0 --stage3-mse-guard-weight 12.0 --grad-clip-norm 1.0 --wandb-mode offline
```

## Summary

- loss_mean: 0.14829637073601284
- image_l1_mean: 0.052545610561501234
- lpips_mean: 0.4967394753607611
- ms_ssim_loss_mean: 0.2515834283828735
- stage3_l1_guard_mean: 1.720416514823834e-05
- stage3_mse_guard_mean: 2.524376439396292e-05
- alpha_mean_loss_mean: 0.0
- stage4_psnr_mean: 22.107007043361662
- stage3_psnr_mean: 22.063367989063263
- stage4_ms_ssim_mean: 0.7484165714929502
- stage3_ms_ssim_mean: 0.7449246131380399
- alpha_mean_mean: 0.29878336588541665
- alpha_std_mean: 0.0
- alpha_min_mean: 0.29878336588541665
- alpha_max_mean: 0.29878336588541665
- alpha_mean_over_steps: 0.29878336588541665
- alpha_mean_std_over_steps: 0.03447498753666878
- payload_policy: deterministic decoder-side gate from decoded CoSER tensors and fixed model weights; no transmitted side information

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_gate/20260628_stage4_cod_lite_gate_statsmatch_fidelity_600_b1ga2.pt`
- summary: `results/stage4_cod_lite_gate/20260628_stage4_cod_lite_gate_statsmatch_fidelity_600_b1ga2/summary.json`
- results_dir: `results/stage4_cod_lite_gate/20260628_stage4_cod_lite_gate_statsmatch_fidelity_600_b1ga2`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_222543-121lz503`
