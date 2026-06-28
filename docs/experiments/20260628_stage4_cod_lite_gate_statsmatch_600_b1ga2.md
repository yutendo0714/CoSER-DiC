# 20260628_stage4_cod_lite_gate_statsmatch_600_b1ga2

Date: 2026-06-28T22:23:53

## Command

```bash
scripts/train_stage4_cod_lite_gate.py --adapter-checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_statsmatch_ft2k_b8.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_traincache_clic_oi_2048_seed20260628/reconstructions/manifest.jsonl --run-name 20260628_stage4_cod_lite_gate_statsmatch_600_b1ga2 --crop-size 512 --batch-size 1 --grad-accum-steps 2 --num-workers 2 --max-steps 600 --lr 1e-4 --hidden-channels 64 --num-blocks 1 --alpha-min 0.0 --alpha-max 0.7 --init-alpha 0.3 --image-l1-weight 0.5 --lpips-weight 0.08 --ms-ssim-weight 0.2 --stage3-l1-guard-weight 2.0 --stage3-mse-guard-weight 4.0 --grad-clip-norm 1.0 --wandb-mode offline
```

## Summary

- loss_mean: 0.11556466348624478
- image_l1_mean: 0.05285645223843555
- lpips_mean: 0.47137488168353836
- ms_ssim_loss_mean: 0.25310693696141245
- stage3_l1_guard_mean: 0.00014811850463350613
- stage3_mse_guard_mean: 0.00012720548188857115
- alpha_mean_loss_mean: 0.0
- stage4_psnr_mean: 22.021042239665984
- stage3_psnr_mean: 22.063367989063263
- stage4_ms_ssim_mean: 0.7468930629392465
- stage3_ms_ssim_mean: 0.7449246131380399
- alpha_mean_mean: 0.42943359375
- alpha_std_mean: 0.0
- alpha_min_mean: 0.42943359375
- alpha_max_mean: 0.42943359375
- alpha_mean_over_steps: 0.42943359375
- alpha_mean_std_over_steps: 0.05059682950377464
- payload_policy: deterministic decoder-side gate from decoded CoSER tensors and fixed model weights; no transmitted side information

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_gate/20260628_stage4_cod_lite_gate_statsmatch_600_b1ga2.pt`
- summary: `results/stage4_cod_lite_gate/20260628_stage4_cod_lite_gate_statsmatch_600_b1ga2/summary.json`
- results_dir: `results/stage4_cod_lite_gate/20260628_stage4_cod_lite_gate_statsmatch_600_b1ga2`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_222245-ee4f1f38`
