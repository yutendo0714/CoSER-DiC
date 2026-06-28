# 20260628_stage4_cod_lite_pyramid_sem256_detailctx6_statsmatch_ft2k_b8

Date: 2026-06-28T21:59:13

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_traincache_clic_oi_2048_seed20260628/reconstructions/manifest.jsonl --run-name 20260628_stage4_cod_lite_pyramid_sem256_detailctx6_statsmatch_ft2k_b8 --crop-size 512 --batch-size 8 --num-workers 4 --max-steps 2000 --lr 5e-5 --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_tanh075_traincache2048_5k_b8.pt --condition-residual-scale 0.75 --condition-residual-tanh --condition-l1-weight 1.0 --condition-cosine-weight 0.25 --condition-channel-stats-weight 0.20 --condition-highfreq-weight 0.05 --image-l1-weight 0.0 --lpips-weight 0.0 --grad-clip-norm 1.0 --save-sample-every 0 --wandb-mode offline
```

## Summary

- loss_mean: 0.4914104907810688
- condition_l1_mean: 0.38752494570612905
- condition_cosine_loss_mean: 0.2136610065102577
- condition_channel_stats_loss_mean: 0.23856755690276624
- condition_highfreq_loss_mean: 0.055135635623708365
- image_l1_mean: 0.0
- lpips_mean: 0.0
- stage4_psnr_mean: 0.0
- stage3_psnr_mean: 22.072579319000244
- condition_residual_l1_mean: 0.34003626741468906
- condition_delta_raw_l1_mean: 0.6508243479132653

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_statsmatch_ft2k_b8.pt`
- summary: `results/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_statsmatch_ft2k_b8/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_statsmatch_ft2k_b8`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_215305-85k6u1sy`
