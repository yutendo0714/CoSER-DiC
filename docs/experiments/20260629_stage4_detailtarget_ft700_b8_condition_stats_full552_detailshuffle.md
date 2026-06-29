# 20260629_stage4_detailtarget_ft700_b8_condition_stats_full552_detailshuffle

Date: 2026-06-29T16:15:10

## Command

```bash
scripts/analyze_stage4_cod_lite_condition_stats.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailtarget_ft700_b8.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl --per-image-metrics results/stage4_cod_lite_adapter_eval/20260629_stage4_detailtarget_ft700_b8_full552_eval/per_image_metrics.jsonl --run-name 20260629_stage4_detailtarget_ft700_b8_condition_stats_full552_detailshuffle --crop-size 512 --batch-size 8 --num-workers 4 --detail-context-ablation shuffle --wandb-mode offline
```

## Summary

- target_mean_mean: -0.008482018300009773
- target_std_mean: 0.7953917812825977
- target_min_mean: -8.458701313405797
- target_max_mean: 8.116932744565217
- target_abs_mean_mean: 0.576335241937119
- target_rms_mean: 0.7957490486079368
- target_l2_mean: 498.9898513572804
- target_channel_mean_std_mean: 0.47448265034219494
- target_channel_std_mean_mean: 0.575138468079377
- target_channel_std_std_mean: 0.2498037982057186
- target_spatial_highfreq_ratio_mean: 0.2574875167692485
- base_mean_mean: -0.04714095981182544
- base_std_mean: 0.8234664921959242
- base_min_mean: -9.861129981884059
- base_max_mean: 9.098137454710145
- base_abs_mean_mean: 0.589333248095236
- base_rms_mean: 0.8251020131983618
- base_l2_mean: 517.3961961718574
- base_channel_mean_std_mean: 0.5554099898191466
- base_channel_std_mean_mean: 0.5356575294897177
- base_channel_std_std_mean: 0.2662208077862211
- base_spatial_highfreq_ratio_mean: 0.27760245748188184
- pred_mean_mean: -0.013785669269516116
- pred_std_mean: 0.7367589554708937
- pred_min_mean: -9.041553442028986
- pred_max_mean: 8.43549026268116
- pred_abs_mean_mean: 0.5170436381105927
- pred_rms_mean: 0.7371502788602442
- pred_l2_mean: 462.2443569017493
- pred_channel_mean_std_mean: 0.4376263230283191
- pred_channel_std_mean_mean: 0.523030041406552
- pred_channel_std_std_mean: 0.25453116276395926
- pred_spatial_highfreq_ratio_mean: 0.2424254982151847
- adapter_delta_raw_mean_mean: 0.06835131588108513
- adapter_delta_raw_std_mean: 0.8084965729950995
- adapter_delta_raw_min_mean: -6.730129076086956
- adapter_delta_raw_max_mean: 9.229789402173912
- adapter_delta_raw_abs_mean_mean: 0.5483211099669554
- adapter_delta_raw_rms_mean: 0.8115086144079333
- adapter_delta_raw_l2_mean: 508.87219293566716
- adapter_delta_raw_channel_mean_std_mean: 0.6475924219990122
- adapter_delta_raw_channel_std_mean_mean: 0.43357001963085023
- adapter_delta_raw_channel_std_std_mean: 0.1995848680212014
- adapter_delta_raw_spatial_highfreq_ratio_mean: 0.23358953773867394
- base_to_target_l1_mean: 0.5373085212988266
- base_to_target_mse_mean: 0.48462129287097766
- base_to_target_rms_mean: 0.6914442249074362
- base_to_target_relative_l2_mean: 0.8762243978031303
- base_to_target_cosine_mean: 0.6218612093235487
- base_to_target_mean_delta_mean: -0.038658941766935524
- base_to_target_std_delta_mean: 0.028074710913326428
- pred_to_target_l1_mean: 0.4112901489479818
- pred_to_target_mse_mean: 0.2835272090860467
- pred_to_target_rms_mean: 0.5299399477848108
- pred_to_target_relative_l2_mean: 0.6707284386823142
- pred_to_target_cosine_mean: 0.7550254401521407
- pred_to_target_mean_delta_mean: -0.005303651038087517
- pred_to_target_std_delta_mean: -0.058632825811704
- pred_to_base_l1_mean: 0.3435436839750711
- pred_to_base_mse_mean: 0.17280639358458744
- pred_to_base_rms_mean: 0.41384198435622715
- pred_to_base_relative_l2_mean: 0.5059616216390893
- pred_to_base_cosine_mean: 0.8599544223667919
- pred_to_base_mean_delta_mean: 0.03335529071467834
- pred_to_base_std_delta_mean: -0.08670753672503043
- pred_l1_improvement_vs_base_mean: 0.1260183723508448
- pred_relative_l2_improvement_vs_base_mean: 0.20549595912081609
- count: 552
- actual_payload_bpp_mean: 0.013999109682829483
- pred_condition_l1_win_rate: 0.9981884057971014
- pred_relative_l2_win_rate: 1.0
- condition_residual_scale: 0.85
- condition_residual_tanh: True
- condition_delta_ablation: normal
- semantic_latent_ablation: normal
- detail_context_ablation: shuffle
- ablation_shuffle_seed: 1234
- payload_policy: analysis only; predicted condition is deterministic from decoded CoSER tensors, fixed adapter weights, and fixed CoD-Lite weights, so no image-specific side information is introduced

## Artifacts

- summary: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailtarget_ft700_b8_condition_stats_full552_detailshuffle/summary.json`
- per_image_condition_stats: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailtarget_ft700_b8_condition_stats_full552_detailshuffle/per_image_condition_stats.jsonl`
- channel_stats: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailtarget_ft700_b8_condition_stats_full552_detailshuffle/channel_stats.json`
- activation_histograms: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailtarget_ft700_b8_condition_stats_full552_detailshuffle/activation_histograms.json`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_161450-ri30ex80`
