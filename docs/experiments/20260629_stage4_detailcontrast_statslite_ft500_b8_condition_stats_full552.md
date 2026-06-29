# 20260629_stage4_detailcontrast_statslite_ft500_b8_condition_stats_full552

Date: 2026-06-29T15:57:45

## Command

```bash
scripts/analyze_stage4_cod_lite_condition_stats.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailcontrast_statslite_ft500_b8.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl --per-image-metrics results/stage4_cod_lite_adapter_eval/20260629_stage4_detailcontrast_statslite_ft500_b8_full552_eval/per_image_metrics.jsonl --run-name 20260629_stage4_detailcontrast_statslite_ft500_b8_condition_stats_full552 --crop-size 512 --batch-size 8 --num-workers 4 --wandb-mode offline
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
- pred_mean_mean: -0.01421319632475164
- pred_std_mean: 0.7456458288690319
- pred_min_mean: -9.053017436594203
- pred_max_mean: 8.451822916666666
- pred_abs_mean_mean: 0.5254419602561688
- pred_rms_mean: 0.7460379888829978
- pred_l2_mean: 467.8175679635311
- pred_channel_mean_std_mean: 0.4377821145606214
- pred_channel_std_mean_mean: 0.5347786637535994
- pred_channel_std_std_mean: 0.25506662302043126
- pred_spatial_highfreq_ratio_mean: 0.24201358358065286
- adapter_delta_raw_mean_mean: 0.0697561301007543
- adapter_delta_raw_std_mean: 0.8445762639352377
- adapter_delta_raw_min_mean: -6.958984375
- adapter_delta_raw_max_mean: 9.784901494565217
- adapter_delta_raw_abs_mean_mean: 0.5692594791560069
- adapter_delta_raw_rms_mean: 0.8475920440278192
- adapter_delta_raw_l2_mean: 531.4990043087282
- adapter_delta_raw_channel_mean_std_mean: 0.6809155231044777
- adapter_delta_raw_channel_std_mean_mean: 0.44552391859284346
- adapter_delta_raw_channel_std_std_mean: 0.20766320059318905
- adapter_delta_raw_spatial_highfreq_ratio_mean: 0.2274397678160365
- base_to_target_l1_mean: 0.5373085212988266
- base_to_target_mse_mean: 0.48462129287097766
- base_to_target_rms_mean: 0.6914442249074362
- base_to_target_relative_l2_mean: 0.8762243978031303
- base_to_target_cosine_mean: 0.6218612093235487
- base_to_target_mean_delta_mean: -0.038658941766935524
- base_to_target_std_delta_mean: 0.028074710913326428
- pred_to_target_l1_mean: 0.41185790165394975
- pred_to_target_mse_mean: 0.2840085067934748
- pred_to_target_rms_mean: 0.5305230622922165
- pred_to_target_relative_l2_mean: 0.6714674642552501
- pred_to_target_cosine_mean: 0.7568129567780356
- pred_to_target_mean_delta_mean: -0.005731178135911118
- pred_to_target_std_delta_mean: -0.04974595241356587
- pred_to_base_l1_mean: 0.3504624669102655
- pred_to_base_mse_mean: 0.17900069094384494
- pred_to_base_rms_mean: 0.4210828428791053
- pred_to_base_relative_l2_mean: 0.5148798519394536
- pred_to_base_cosine_mean: 0.855215928044872
- pred_to_base_mean_delta_mean: 0.03292776359181068
- pred_to_base_std_delta_mean: -0.0778206633268923
- pred_l1_improvement_vs_base_mean: 0.12545061964487683
- pred_relative_l2_improvement_vs_base_mean: 0.20475693354788033
- count: 552
- actual_payload_bpp_mean: 0.013999109682829483
- pred_condition_l1_win_rate: 1.0
- pred_relative_l2_win_rate: 1.0
- condition_residual_scale: 0.85
- condition_residual_tanh: True
- condition_delta_ablation: normal
- semantic_latent_ablation: normal
- detail_context_ablation: normal
- ablation_shuffle_seed: 1234
- payload_policy: analysis only; predicted condition is deterministic from decoded CoSER tensors, fixed adapter weights, and fixed CoD-Lite weights, so no image-specific side information is introduced

## Artifacts

- summary: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailcontrast_statslite_ft500_b8_condition_stats_full552/summary.json`
- per_image_condition_stats: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailcontrast_statslite_ft500_b8_condition_stats_full552/per_image_condition_stats.jsonl`
- channel_stats: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailcontrast_statslite_ft500_b8_condition_stats_full552/channel_stats.json`
- activation_histograms: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailcontrast_statslite_ft500_b8_condition_stats_full552/activation_histograms.json`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_155724-0e29fy7h`
