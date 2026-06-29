# 20260629_stage4_fast8192_semdrop_smoke1_b4

Date: 2026-06-29T01:49:00

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage4_fast8192_semdrop_smoke1_b4 --crop-size 512 --batch-size 4 --grad-accum-steps 1 --num-workers 2 --max-steps 1 --lr 1e-5 --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailaware_ft600_fast8192_ft1500_b4.pt --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 0.80 --condition-cosine-weight 0.25 --condition-channel-stats-weight 0.20 --condition-highfreq-weight 0.05 --image-l1-weight 0.08 --lpips-weight 0.03 --ms-ssim-weight 0.08 --stage3-l1-guard-weight 0.60 --stage3-mse-guard-weight 1.20 --semantic-latent-dropout-prob 0.25 --detail-context-dropout-prob 0.00 --grad-clip-norm 1.0 --save-sample-every 0 --wandb-mode offline
```

## Summary

- loss_mean: 0.4874770939350128
- condition_l1_mean: 0.3977084159851074
- condition_cosine_loss_mean: 0.25669145584106445
- condition_channel_stats_loss_mean: 0.25977465510368347
- condition_highfreq_loss_mean: 0.06150355562567711
- base_condition_l1_mean: 0.4977138936519623
- condition_l1_delta_vs_base_mean: -0.10000547766685486
- condition_cosine_mean: 0.7433085441589355
- pred_condition_std_mean: 0.6786532402038574
- target_condition_std_mean: 0.7573367357254028
- pred_condition_highfreq_mean: 0.25161290168762207
- target_condition_highfreq_mean: 0.2775115370750427
- image_l1_mean: 0.0587816946208477
- lpips_mean: 0.4013741612434387
- dists_loss_mean: 0.0
- ms_ssim_loss_mean: 0.32702624797821045
- stage3_l1_guard_mean: 0.008693856187164783
- stage3_mse_guard_mean: 0.0016543806996196508
- stage4_psnr_mean: 21.645179748535156
- stage3_psnr_mean: 22.623947143554688
- condition_residual_l1_mean: 0.3397398889064789
- condition_delta_raw_l1_mean: 0.5296127200126648
- semantic_latent_drop_fraction_mean: 0.25
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage4_fast8192_semdrop_smoke1_b4.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage4_fast8192_semdrop_smoke1_b4/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage4_fast8192_semdrop_smoke1_b4`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_014857-p5z4blhg`
