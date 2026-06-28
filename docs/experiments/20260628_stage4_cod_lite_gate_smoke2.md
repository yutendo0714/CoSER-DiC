# 20260628_stage4_cod_lite_gate_smoke2

Date: 2026-06-28T22:21:37

## Command

```bash
scripts/train_stage4_cod_lite_gate.py --adapter-checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_statsmatch_ft2k_b8.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_traincache_clic_oi_2048_seed20260628/reconstructions/manifest.jsonl --run-name 20260628_stage4_cod_lite_gate_smoke2 --crop-size 512 --limit 8 --batch-size 1 --num-workers 0 --max-steps 2 --lr 1e-4 --hidden-channels 32 --num-blocks 1 --init-alpha 0.3 --image-l1-weight 1.0 --lpips-weight 0.0 --ms-ssim-weight 0.1 --stage3-l1-guard-weight 1.0 --stage3-mse-guard-weight 1.0 --wandb-mode offline
```

## Summary

- loss_mean: 0.07229042984545231
- image_l1_mean: 0.044628411531448364
- lpips_mean: 0.0
- ms_ssim_loss_mean: 0.2765788435935974
- stage3_l1_guard_mean: 0.0
- stage3_mse_guard_mean: 4.133675247430801e-06
- alpha_mean_loss_mean: 0.0
- stage4_psnr_mean: 24.57975196838379
- stage3_psnr_mean: 24.574514389038086
- stage4_ms_ssim_mean: 0.7234211564064026
- stage3_ms_ssim_mean: 0.7209230363368988
- alpha_mean_mean: 0.30078125
- alpha_std_mean: 0.0
- alpha_min_mean: 0.30078125
- alpha_max_mean: 0.30078125
- alpha_mean_over_steps: 0.30078125
- alpha_mean_std_over_steps: 0.0
- payload_policy: deterministic decoder-side gate from decoded CoSER tensors and fixed model weights; no transmitted side information

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_gate/20260628_stage4_cod_lite_gate_smoke2.pt`
- summary: `results/stage4_cod_lite_gate/20260628_stage4_cod_lite_gate_smoke2/summary.json`
- results_dir: `results/stage4_cod_lite_gate/20260628_stage4_cod_lite_gate_smoke2`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_222136-s8zlgeko`
