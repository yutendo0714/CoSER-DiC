# 20260628_stage4_detailaware_adapter_idstart_ft600_b2

Date: 2026-06-28T23:44:04

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_traincache_clic_oi_2048_seed20260628/reconstructions/manifest.jsonl --run-name 20260628_stage4_detailaware_adapter_idstart_ft600_b2 --crop-size 512 --batch-size 2 --grad-accum-steps 1 --num-workers 2 --max-steps 600 --lr 2e-5 --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-detail-blocks 3 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_statsmatch_ft2k_b8.pt --init-nonstrict --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 1.0 --condition-cosine-weight 0.25 --condition-channel-stats-weight 0.20 --condition-highfreq-weight 0.05 --image-l1-weight 0.08 --lpips-weight 0.03 --ms-ssim-weight 0.08 --stage3-l1-guard-weight 0.60 --stage3-mse-guard-weight 1.20 --grad-clip-norm 1.0 --save-sample-every 200 --wandb-mode offline
```

## Summary

- loss_mean: 0.5206955995659034
- condition_l1_mean: 0.37997979114452995
- condition_cosine_loss_mean: 0.20423873881498972
- condition_channel_stats_loss_mean: 0.2208189099530379
- condition_highfreq_loss_mean: 0.05427485015243292
- base_condition_l1_mean: 0.5174275702734789
- condition_l1_delta_vs_base_mean: -0.13744777912894884
- condition_cosine_mean: 0.7957612611850102
- pred_condition_std_mean: 0.7502100880940755
- target_condition_std_mean: 0.8100000576178232
- pred_condition_highfreq_mean: 0.2198586622128884
- target_condition_highfreq_mean: 0.2439681992183129
- image_l1_mean: 0.05747675898484886
- lpips_mean: 0.39124565427501995
- ms_ssim_loss_mean: 0.27718911786874134
- stage3_l1_guard_mean: 0.004275457865248124
- stage3_mse_guard_mean: 0.001418903800464856
- stage4_psnr_mean: 21.31579472541809
- stage3_psnr_mean: 22.125426925023397
- condition_residual_l1_mean: 0.35927539204557735
- condition_delta_raw_l1_mean: 0.5800484045346578

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260628_stage4_detailaware_adapter_idstart_ft600_b2.pt`
- summary: `results/stage4_cod_lite_adapter/20260628_stage4_detailaware_adapter_idstart_ft600_b2/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260628_stage4_detailaware_adapter_idstart_ft600_b2`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_234237-nl1ppntz`
