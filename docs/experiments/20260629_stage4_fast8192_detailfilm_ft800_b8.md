# 20260629_stage4_fast8192_detailfilm_ft800_b8

Date: 2026-06-29T03:16:46

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage4_fast8192_detailfilm_ft800_b8 --crop-size 512 --batch-size 8 --num-workers 4 --max-steps 800 --lr 5e-6 --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --detail-control-branch --detail-film-modulation --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailcontrol_semdrop020_ft800_b8.pt --init-nonstrict --semantic-latent-dropout-prob 0.20 --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 1.0 --condition-cosine-weight 0.20 --condition-channel-stats-weight 0.15 --condition-highfreq-weight 0.03 --image-l1-weight 0.25 --lpips-weight 0.03 --ms-ssim-weight 0.02 --stage3-l1-guard-weight 0.20 --stage3-mse-guard-weight 0.05 --grad-clip-norm 1.0 --save-sample-every 200 --wandb-mode offline
```

## Summary

- loss_mean: 0.5130129995569587
- condition_l1_mean: 0.39496012452989815
- condition_cosine_loss_mean: 0.22185062937438488
- condition_channel_stats_loss_mean: 0.25530868537724016
- condition_highfreq_loss_mean: 0.056105864932760596
- detail_contrast_loss_mean: 0.0
- detail_condition_l1_zero_mean: 0.0
- detail_condition_l1_gap_mean: 0.0
- detail_highfreq_residual_loss_mean: 0.0
- detail_highfreq_residual_pred_l1_mean: 0.0
- detail_highfreq_residual_target_l1_mean: 0.0
- base_condition_l1_mean: 0.5171486399695278
- condition_l1_delta_vs_base_mean: -0.12218851543962955
- condition_cosine_mean: 0.7781493706256151
- pred_condition_std_mean: 0.732540303170681
- target_condition_std_mean: 0.8105030023306609
- pred_condition_highfreq_mean: 0.22337907515466213
- target_condition_highfreq_mean: 0.24475151481106877
- image_l1_mean: 0.0593622018257156
- lpips_mean: 0.4046007274836302
- dists_loss_mean: 0.0
- ms_ssim_loss_mean: 0.27996017813682555
- stage3_l1_guard_mean: 0.005225353418791201
- stage3_mse_guard_mean: 0.0016084673393925186
- stage4_psnr_mean: 21.077968916893006
- stage3_psnr_mean: 21.963221395015715
- condition_residual_l1_mean: 0.33639640651643277
- condition_delta_raw_l1_mean: 0.5262340184301137
- semantic_latent_drop_fraction_mean: 0.20140625
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailfilm_ft800_b8.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailfilm_ft800_b8/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailfilm_ft800_b8`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_030949-z164jpvn`
