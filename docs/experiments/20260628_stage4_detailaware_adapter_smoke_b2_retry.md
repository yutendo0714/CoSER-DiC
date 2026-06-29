# 20260628_stage4_detailaware_adapter_smoke_b2_retry

Date: 2026-06-28T23:41:38

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_traincache_clic_oi_2048_seed20260628/reconstructions/manifest.jsonl --run-name 20260628_stage4_detailaware_adapter_smoke_b2_retry --crop-size 512 --limit 8 --batch-size 2 --grad-accum-steps 1 --num-workers 2 --max-steps 3 --lr 2e-5 --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-detail-blocks 3 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_statsmatch_ft2k_b8.pt --init-nonstrict --condition-residual-scale 0.75 --condition-residual-tanh --condition-l1-weight 1.0 --condition-cosine-weight 0.25 --condition-channel-stats-weight 0.20 --condition-highfreq-weight 0.05 --image-l1-weight 0.10 --ms-ssim-weight 0.10 --stage3-l1-guard-weight 0.75 --stage3-mse-guard-weight 1.50 --grad-clip-norm 1.0 --save-sample-every 1 --wandb-mode offline
```

## Summary

- loss_mean: 0.5277912418047587
- condition_l1_mean: 0.38626397649447125
- condition_cosine_loss_mean: 0.221332848072052
- condition_channel_stats_loss_mean: 0.224671537677447
- condition_highfreq_loss_mean: 0.05056849122047424
- base_condition_l1_mean: 0.5162877937157949
- condition_l1_delta_vs_base_mean: -0.13002381722132364
- condition_cosine_mean: 0.778667151927948
- pred_condition_std_mean: 0.7235797246297201
- target_condition_std_mean: 0.7874974608421326
- pred_condition_highfreq_mean: 0.2203808675209681
- target_condition_highfreq_mean: 0.23866362373034158
- image_l1_mean: 0.053833503276109695
- lpips_mean: 0.0
- ms_ssim_loss_mean: 0.2721313238143921
- stage3_l1_guard_mean: 0.00543608795851469
- stage3_mse_guard_mean: 0.0013718486491901178
- stage4_psnr_mean: 22.16756312052409
- stage3_psnr_mean: 23.167856216430664
- condition_residual_l1_mean: 0.3454756736755371
- condition_delta_raw_l1_mean: 0.6726172765096029

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260628_stage4_detailaware_adapter_smoke_b2_retry.pt`
- summary: `results/stage4_cod_lite_adapter/20260628_stage4_detailaware_adapter_smoke_b2_retry/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260628_stage4_detailaware_adapter_smoke_b2_retry`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_234136-nhkvnrx6`
