# 20260629_stage4_detailtarget_perceptual_ft700_b8_condition_stats_full552

Date: 2026-06-29T16:28:33

## Command

```bash
scripts/analyze_stage4_cod_lite_condition_stats.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailtarget_perceptual_ft700_b8.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl --per-image-metrics results/stage4_cod_lite_adapter_eval/20260629_stage4_detailtarget_perceptual_ft700_b8_full552_eval/per_image_metrics.jsonl --run-name 20260629_stage4_detailtarget_perceptual_ft700_b8_condition_stats_full552 --crop-size 512 --batch-size 8 --num-workers 4 --wandb-mode offline
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
- pred_mean_mean: -0.014165153023671904
- pred_std_mean: 0.7376783644591552
- pred_min_mean: -9.04107223731884
- pred_max_mean: 8.431640625
- pred_abs_mean_mean: 0.5179596822978794
- pred_rms_mean: 0.7380757518652556
- pred_l2_mean: 462.8246932098831
- pred_channel_mean_std_mean: 0.4373185746911643
- pred_channel_std_mean_mean: 0.5244464385660662
- pred_channel_std_std_mean: 0.2548383213849603
- pred_spatial_highfreq_ratio_mean: 0.2436090674035359
- adapter_delta_raw_mean_mean: 0.07036627759225667
- adapter_delta_raw_std_mean: 0.8510654925004296
- adapter_delta_raw_min_mean: -7.111271512681159
- adapter_delta_raw_max_mean: 10.015908061594203
- adapter_delta_raw_abs_mean_mean: 0.5725957865192406
- adapter_delta_raw_rms_mean: 0.8541049122594405
- adapter_delta_raw_l2_mean: 535.583024729853
- adapter_delta_raw_channel_mean_std_mean: 0.6858905755739281
- adapter_delta_raw_channel_std_mean_mean: 0.44845772815355356
- adapter_delta_raw_channel_std_std_mean: 0.2123947333868431
- adapter_delta_raw_spatial_highfreq_ratio_mean: 0.22999459349860749
- base_to_target_l1_mean: 0.5373085212988266
- base_to_target_mse_mean: 0.48462129287097766
- base_to_target_rms_mean: 0.6914442249074362
- base_to_target_relative_l2_mean: 0.8762243978031303
- base_to_target_cosine_mean: 0.6218612093235487
- base_to_target_mean_delta_mean: -0.038658941766935524
- base_to_target_std_delta_mean: 0.028074710913326428
- pred_to_target_l1_mean: 0.4102275326200154
- pred_to_target_mse_mean: 0.281898968839559
- pred_to_target_rms_mean: 0.5285514985629614
- pred_to_target_relative_l2_mean: 0.6689681194830632
- pred_to_target_cosine_mean: 0.7567158632952234
- pred_to_target_mean_delta_mean: -0.005683134801881971
- pred_to_target_std_delta_mean: -0.05771341682344243
- pred_to_base_l1_mean: 0.35177392409979436
- pred_to_base_mse_mean: 0.18013850915367188
- pred_to_base_rms_mean: 0.42243127578842465
- pred_to_base_relative_l2_mean: 0.5165695010859898
- pred_to_base_cosine_mean: 0.8537011882965115
- pred_to_base_mean_delta_mean: 0.03297580672475253
- pred_to_base_std_delta_mean: -0.08578812773676886
- pred_l1_improvement_vs_base_mean: 0.12708098867881126
- pred_relative_l2_improvement_vs_base_mean: 0.20725627832006718
- count: 552
- actual_payload_bpp_mean: 0.013999109682829483
- pred_condition_l1_win_rate: 0.9981884057971014
- pred_relative_l2_win_rate: 1.0
- condition_residual_scale: 0.85
- condition_residual_tanh: True
- condition_delta_ablation: normal
- semantic_latent_ablation: normal
- detail_context_ablation: normal
- ablation_shuffle_seed: 1234
- payload_policy: analysis only; predicted condition is deterministic from decoded CoSER tensors, fixed adapter weights, and fixed CoD-Lite weights, so no image-specific side information is introduced

## Artifacts

- summary: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailtarget_perceptual_ft700_b8_condition_stats_full552/summary.json`
- per_image_condition_stats: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailtarget_perceptual_ft700_b8_condition_stats_full552/per_image_condition_stats.jsonl`
- channel_stats: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailtarget_perceptual_ft700_b8_condition_stats_full552/channel_stats.json`
- activation_histograms: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailtarget_perceptual_ft700_b8_condition_stats_full552/activation_histograms.json`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_162812-vbqajth4`
