# 20260628_stage4_cod_lite_condition_gate_fidelity_probe300_b1ga2

Date: 2026-06-28T23:02:56

## Command

```bash
scripts/train_stage4_cod_lite_condition_gate.py --adapter-checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_statsmatch_ft2k_b8.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_traincache_clic_oi_2048_seed20260628/reconstructions/manifest.jsonl --run-name 20260628_stage4_cod_lite_condition_gate_fidelity_probe300_b1ga2 --crop-size 512 --batch-size 1 --grad-accum-steps 2 --num-workers 2 --max-steps 300 --lr 1e-4 --hidden-channels 64 --num-blocks 1 --gate-max 0.55 --init-gate 0.2 --image-l1-weight 1.5 --lpips-weight 0.04 --ms-ssim-weight 0.3 --condition-l1-weight 0.03 --stage3-l1-guard-weight 8.0 --stage3-mse-guard-weight 16.0 --gate-mean-target 0.2 --gate-mean-weight 0.05 --grad-clip-norm 1.0 --wandb-mode offline
```

## Summary

- loss_mean: 0.26850978571921585
- image_l1_mean: 0.05889675233202676
- lpips_mean: 0.4969164730111758
- ms_ssim_loss_mean: 0.280350108842055
- condition_l1_mean: 0.4359015390276909
- ungated_condition_l1_mean: 0.38215898983180524
- condition_regression_guard_mean: 0.053859035496910415
- stage3_l1_guard_mean: 0.004882167171066006
- stage3_mse_guard_mean: 0.0013167349905415903
- gate_mean_loss_mean: 0.059616420935184704
- stage4_psnr_mean: 21.31764971415202
- stage3_psnr_mean: 22.110201001167297
- stage4_ms_ssim_mean: 0.7196498908102512
- stage3_ms_ssim_mean: 0.7412123055259386
- gate_mean_mean: 0.42852698008219403
- gate_std_mean: 0.04579637418868515
- gate_min_mean: 0.25763509114583333
- gate_max_mean: 0.49808919270833335
- gate_mean_over_steps: 0.42852698008219403
- gate_mean_std_over_steps: 0.08357331156730652
- payload_policy: deterministic decoder-side condition gate from decoded CoSER tensors and fixed model weights; no transmitted side information and no RGB output blending

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_condition_gate/20260628_stage4_cod_lite_condition_gate_fidelity_probe300_b1ga2.pt`
- summary: `results/stage4_cod_lite_condition_gate/20260628_stage4_cod_lite_condition_gate_fidelity_probe300_b1ga2/summary.json`
- results_dir: `results/stage4_cod_lite_condition_gate/20260628_stage4_cod_lite_condition_gate_fidelity_probe300_b1ga2`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_230202-lgl77u4r`
