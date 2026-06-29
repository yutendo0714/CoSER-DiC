# 20260629_stage4_fast8192_detailcontrol_semdrop020_ft800_b8

Date: 2026-06-29T02:19:52

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage4_fast8192_detailcontrol_semdrop020_ft800_b8 --crop-size 512 --batch-size 8 --num-workers 4 --max-steps 800 --lr 5e-6 --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --detail-control-branch --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailcontrol_ft800_b4.pt --semantic-latent-dropout-prob 0.20 --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 1.0 --condition-cosine-weight 0.20 --condition-channel-stats-weight 0.15 --condition-highfreq-weight 0.03 --image-l1-weight 0.25 --lpips-weight 0.03 --ms-ssim-weight 0.02 --stage3-l1-guard-weight 0.20 --stage3-mse-guard-weight 0.05 --grad-clip-norm 1.0 --save-sample-every 200 --wandb-mode offline
```

## Summary

- loss_mean: 0.5155408160760999
- condition_l1_mean: 0.3965431332960725
- condition_cosine_loss_mean: 0.22393687710165977
- condition_channel_stats_loss_mean: 0.2573628658428788
- condition_highfreq_loss_mean: 0.05625602509826422
- base_condition_l1_mean: 0.5168800438940525
- condition_l1_delta_vs_base_mean: -0.12033691059798002
- condition_cosine_mean: 0.7760631228983402
- pred_condition_std_mean: 0.7306021162122488
- target_condition_std_mean: 0.8098107255250215
- pred_condition_highfreq_mean: 0.22450986612588167
- target_condition_highfreq_mean: 0.24497095528990032
- image_l1_mean: 0.05978840767405927
- lpips_mean: 0.40491788636893034
- dists_loss_mean: 0.0
- ms_ssim_loss_mean: 0.27955198496580125
- stage3_l1_guard_mean: 0.005754754374793265
- stage3_mse_guard_mean: 0.001631289010620094
- stage4_psnr_mean: 21.080732276439665
- stage3_psnr_mean: 21.993137352466583
- condition_residual_l1_mean: 0.3373467424884439
- condition_delta_raw_l1_mean: 0.5271242166683078
- semantic_latent_drop_fraction_mean: 0.20140625
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailcontrol_semdrop020_ft800_b8.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailcontrol_semdrop020_ft800_b8/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailcontrol_semdrop020_ft800_b8`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_021255-0kio5enb`
