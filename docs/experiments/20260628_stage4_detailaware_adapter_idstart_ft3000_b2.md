# 20260628_stage4_detailaware_adapter_idstart_ft3000_b2

Date: 2026-06-29T00:03:47

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_traincache_clic_oi_2048_seed20260628/reconstructions/manifest.jsonl --run-name 20260628_stage4_detailaware_adapter_idstart_ft3000_b2 --crop-size 512 --batch-size 2 --grad-accum-steps 1 --num-workers 2 --max-steps 3000 --lr 1e-5 --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-detail-blocks 3 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_detailaware_adapter_idstart_ft600_b2.pt --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 1.0 --condition-cosine-weight 0.25 --condition-channel-stats-weight 0.20 --condition-highfreq-weight 0.05 --image-l1-weight 0.08 --lpips-weight 0.03 --ms-ssim-weight 0.08 --stage3-l1-guard-weight 0.60 --stage3-mse-guard-weight 1.20 --grad-clip-norm 1.0 --save-sample-every 1000 --wandb-mode offline
```

## Summary

- loss_mean: 0.5144903751512369
- condition_l1_mean: 0.3771240880240997
- condition_cosine_loss_mean: 0.20110292122761408
- condition_channel_stats_loss_mean: 0.21411207006374994
- condition_highfreq_loss_mean: 0.0534081943569084
- base_condition_l1_mean: 0.5175407055020332
- condition_l1_delta_vs_base_mean: -0.14041661747793358
- condition_cosine_mean: 0.7988970787723859
- pred_condition_std_mean: 0.751493662873904
- target_condition_std_mean: 0.8103685661753018
- pred_condition_highfreq_mean: 0.21912775274614493
- target_condition_highfreq_mean: 0.24307152704397839
- image_l1_mean: 0.05696979650358359
- lpips_mean: 0.38622263737022877
- ms_ssim_loss_mean: 0.2732901744842529
- stage3_l1_guard_mean: 0.003547361869364977
- stage3_mse_guard_mean: 0.0012181991350371391
- stage4_psnr_mean: 21.391319632212323
- stage3_psnr_mean: 22.077724328041075
- condition_residual_l1_mean: 0.3580696803331375
- condition_delta_raw_l1_mean: 0.5755662910342216

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260628_stage4_detailaware_adapter_idstart_ft3000_b2.pt`
- summary: `results/stage4_cod_lite_adapter/20260628_stage4_detailaware_adapter_idstart_ft3000_b2/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260628_stage4_detailaware_adapter_idstart_ft3000_b2`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_235643-qy47x542`
