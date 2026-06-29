# 20260629_stage5_detailfilm_lora_h256_guarded_denoiser_tail_r4_lr2em05_ft800

Date: 2026-06-29T10:37:10

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailfilm_ft800_b8.pt --run-name 20260629_stage5_detailfilm_lora_h256_guarded_denoiser_tail_r4_lr2em05_ft800 --output-dir checkpoints/stage4_cod_lite_adapter --results-dir results/stage4_cod_lite_adapter --cod-lite-checkpoint external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0312.pt --cod-lite-config external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0312.yaml --crop-size 512 --batch-size 4 --grad-accum-steps 2 --num-workers 4 --max-steps 800 --lr 0.0002 --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --condition-l1-weight 0.25 --condition-cosine-weight 0.1 --condition-channel-stats-weight 0.1 --image-l1-weight 0.18 --lpips-weight 0.04 --dists-weight 0.04 --ms-ssim-weight 0.08 --stage3-l1-guard-weight 2.0 --stage3-mse-guard-weight 4.0 --condition-residual-scale 0.85 --condition-residual-rms-guard-weight 0.05 --condition-residual-rms-guard-ratio 0.35 --condition-residual-rms-guard-granularity channel --backbone-lora-plan results/stage5_inspect/20260629_cod_lite_bpp0312_lora_target_plan_rank4.json --backbone-lora-preset denoiser_tail --backbone-lora-rank 4 --backbone-lora-alpha 4.0 --backbone-lora-lr 2e-05 --backbone-lora-weight-decay 0.0 --grad-clip-norm 1.0 --wandb-mode offline --init-nonstrict --detail-control-branch --detail-film-modulation --condition-residual-tanh --semantic-latent-dropout-prob 0.2
```

## Summary

- loss_mean: 0.2217463277094066
- condition_l1_mean: 0.438601469155401
- condition_cosine_loss_mean: 0.22582359258085488
- condition_channel_stats_loss_mean: 0.2257822247594595
- condition_highfreq_loss_mean: 0.0
- condition_residual_rms_guard_loss_mean: 0.03270056837180164
- condition_residual_rms_ratio_mean_mean: 0.4397858448140323
- condition_residual_rms_ratio_max_mean: 1.2944103626161814
- detail_contrast_loss_mean: 0.0
- detail_condition_l1_zero_mean: 0.0
- detail_condition_l1_gap_mean: 0.0
- detail_highfreq_residual_loss_mean: 0.0
- detail_highfreq_residual_pred_l1_mean: 0.0
- detail_highfreq_residual_target_l1_mean: 0.0
- base_condition_l1_mean: 0.5679146710783243
- condition_l1_delta_vs_base_mean: -0.12931320192292334
- condition_cosine_mean: 0.7741764074191451
- pred_condition_std_mean: 0.8362036104872823
- target_condition_std_mean: 0.8706509080529213
- pred_condition_highfreq_mean: 0.2687462582811713
- target_condition_highfreq_mean: 0.31628550751134754
- image_l1_mean: 0.05560105627402663
- lpips_mean: 0.4400276271253824
- dists_loss_mean: 0.29929573339410126
- ms_ssim_loss_mean: 0.2594317528232932
- stage3_l1_guard_mean: 0.001560656880174065
- stage3_mse_guard_mean: 0.0004608428687333799
- stage4_psnr_mean: 21.675887680053712
- stage3_psnr_mean: 21.94621452450752
- condition_residual_l1_mean: 0.3409613546170294
- condition_delta_raw_l1_mean: 0.5245584266260266
- semantic_latent_drop_fraction_mean: 0.1984375
- detail_context_drop_fraction_mean: 0.0

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage5_detailfilm_lora_h256_guarded_denoiser_tail_r4_lr2em05_ft800.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage5_detailfilm_lora_h256_guarded_denoiser_tail_r4_lr2em05_ft800/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage5_detailfilm_lora_h256_guarded_denoiser_tail_r4_lr2em05_ft800`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_102733-uvuz2rfq`
