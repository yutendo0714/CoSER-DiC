# 20260628_stage4_detailaware_adapter_bpp0078_transfer_ft600_b2

Date: 2026-06-29T00:16:33

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --cod-lite-checkpoint external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0078.pt --cod-lite-config external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0078.yaml --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_traincache_clic_oi_2048_seed20260628/reconstructions/manifest.jsonl --run-name 20260628_stage4_detailaware_adapter_bpp0078_transfer_ft600_b2 --crop-size 512 --batch-size 2 --num-workers 2 --max-steps 600 --lr 1e-5 --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_detailaware_adapter_idstart_ft600_b2.pt --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 1.0 --condition-cosine-weight 0.25 --condition-channel-stats-weight 0.20 --condition-highfreq-weight 0.05 --image-l1-weight 0.08 --lpips-weight 0.03 --ms-ssim-weight 0.08 --stage3-l1-guard-weight 0.60 --stage3-mse-guard-weight 1.20 --wandb-mode offline
```

## Summary

- loss_mean: 0.6049411615232626
- condition_l1_mean: 0.4418585646649202
- condition_cosine_loss_mean: 0.22630185425281524
- condition_channel_stats_loss_mean: 0.2754778246829907
- condition_highfreq_loss_mean: 0.03666724003851414
- base_condition_l1_mean: 0.5209459577004115
- condition_l1_delta_vs_base_mean: -0.07908739303549131
- condition_cosine_mean: 0.7736981457471848
- pred_condition_std_mean: 0.8336432062586149
- target_condition_std_mean: 0.8753949001431465
- pred_condition_highfreq_mean: 0.16233251608908175
- target_condition_highfreq_mean: 0.15313065802057585
- image_l1_mean: 0.06169791202992201
- lpips_mean: 0.41388170411189396
- ms_ssim_loss_mean: 0.30356595138708753
- stage3_l1_guard_mean: 0.008426376160544653
- stage3_mse_guard_mean: 0.0024040165305389866
- stage4_psnr_mean: 20.827785936991372
- stage3_psnr_mean: 22.133621088663737
- condition_residual_l1_mean: 0.3013423799475034
- condition_delta_raw_l1_mean: 0.4353166523079077

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260628_stage4_detailaware_adapter_bpp0078_transfer_ft600_b2.pt`
- summary: `results/stage4_cod_lite_adapter/20260628_stage4_detailaware_adapter_bpp0078_transfer_ft600_b2/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260628_stage4_detailaware_adapter_bpp0078_transfer_ft600_b2`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_001508-ukvz23au`
