# 20260628_stage4_detailaware_adapter_bpp0312_transfer_ft600_b2

Date: 2026-06-29T00:18:51

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --cod-lite-checkpoint external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0312.pt --cod-lite-config external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0312.yaml --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_traincache_clic_oi_2048_seed20260628/reconstructions/manifest.jsonl --run-name 20260628_stage4_detailaware_adapter_bpp0312_transfer_ft600_b2 --crop-size 512 --batch-size 2 --num-workers 2 --max-steps 600 --lr 1e-5 --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_detailaware_adapter_idstart_ft600_b2.pt --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 1.0 --condition-cosine-weight 0.25 --condition-channel-stats-weight 0.20 --condition-highfreq-weight 0.05 --image-l1-weight 0.08 --lpips-weight 0.03 --ms-ssim-weight 0.08 --stage3-l1-guard-weight 0.60 --stage3-mse-guard-weight 1.20 --wandb-mode offline
```

## Summary

- loss_mean: 0.6191926973064741
- condition_l1_mean: 0.4618222302695115
- condition_cosine_loss_mean: 0.2480716636776924
- condition_channel_stats_loss_mean: 0.24765901843706767
- condition_highfreq_loss_mean: 0.06800134050349395
- base_condition_l1_mean: 0.5687378088633219
- condition_l1_delta_vs_base_mean: -0.1069155785938104
- condition_cosine_mean: 0.7519283363223076
- pred_condition_std_mean: 0.8368903378645579
- target_condition_std_mean: 0.867434207101663
- pred_condition_highfreq_mean: 0.27945237894852953
- target_condition_highfreq_mean: 0.31516614568730195
- image_l1_mean: 0.057071260455995795
- lpips_mean: 0.4327255464096864
- ms_ssim_loss_mean: 0.270113470852375
- stage3_l1_guard_mean: 0.003210260084209343
- stage3_mse_guard_mean: 0.0011149824754102156
- stage4_psnr_mean: 21.43160068511963
- stage3_psnr_mean: 22.04689812819163
- condition_residual_l1_mean: 0.3369050760567188
- condition_delta_raw_l1_mean: 0.4995050390561422

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260628_stage4_detailaware_adapter_bpp0312_transfer_ft600_b2.pt`
- summary: `results/stage4_cod_lite_adapter/20260628_stage4_detailaware_adapter_bpp0312_transfer_ft600_b2/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260628_stage4_detailaware_adapter_bpp0312_transfer_ft600_b2`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_001723-bmgj33ae`
