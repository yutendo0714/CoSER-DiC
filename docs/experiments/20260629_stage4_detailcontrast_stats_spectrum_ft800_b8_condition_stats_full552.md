# 20260629_stage4_detailcontrast_stats_spectrum_ft800_b8_condition_stats_full552

Date: 2026-06-29T15:49:25

## Command

```bash
scripts/analyze_stage4_cod_lite_condition_stats.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailcontrast_stats_spectrum_ft800_b8.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl --per-image-metrics results/stage4_cod_lite_adapter_eval/20260629_stage4_detailcontrast_stats_spectrum_ft800_b8_full552_eval/per_image_metrics.jsonl --run-name 20260629_stage4_detailcontrast_stats_spectrum_ft800_b8_condition_stats_full552 --crop-size 512 --batch-size 8 --num-workers 4 --wandb-mode offline
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
- pred_mean_mean: -0.013435536311506907
- pred_std_mean: 0.7556725774986156
- pred_min_mean: -9.052224864130435
- pred_max_mean: 8.45708786231884
- pred_abs_mean_mean: 0.5353262955727784
- pred_rms_mean: 0.7560399826885997
- pred_l2_mean: 474.08951093148494
- pred_channel_mean_std_mean: 0.4371548006715982
- pred_channel_std_mean_mean: 0.5497648170136887
- pred_channel_std_std_mean: 0.2533225019502899
- pred_spatial_highfreq_ratio_mean: 0.24678663931031158
- adapter_delta_raw_mean_mean: 0.07349141303827797
- adapter_delta_raw_std_mean: 0.8890317338316337
- adapter_delta_raw_min_mean: -7.294865262681159
- adapter_delta_raw_max_mean: 10.538439764492754
- adapter_delta_raw_abs_mean_mean: 0.5948809647689695
- adapter_delta_raw_rms_mean: 0.8922107459950275
- adapter_delta_raw_l2_mean: 559.4780264591825
- adapter_delta_raw_channel_mean_std_mean: 0.717069595836211
- adapter_delta_raw_channel_std_mean_mean: 0.46671096807804663
- adapter_delta_raw_channel_std_std_mean: 0.22127185868558244
- adapter_delta_raw_spatial_highfreq_ratio_mean: 0.21560642596982096
- base_to_target_l1_mean: 0.5373085212988266
- base_to_target_mse_mean: 0.48462129287097766
- base_to_target_rms_mean: 0.6914442249074362
- base_to_target_relative_l2_mean: 0.8762243978031303
- base_to_target_cosine_mean: 0.6218612093235487
- base_to_target_mean_delta_mean: -0.038658941766935524
- base_to_target_std_delta_mean: 0.028074710913326428
- pred_to_target_l1_mean: 0.4163469859547373
- pred_to_target_mse_mean: 0.289768461893866
- pred_to_target_rms_mean: 0.5359283715918444
- pred_to_target_relative_l2_mean: 0.6783251602580582
- pred_to_target_cosine_mean: 0.7545734030613
- pred_to_target_mean_delta_mean: -0.004953518031602405
- pred_to_target_std_delta_mean: -0.03971920378398204
- pred_to_base_l1_mean: 0.35871993329214014
- pred_to_base_mse_mean: 0.18650348637037087
- pred_to_base_rms_mean: 0.42968379015075986
- pred_to_base_relative_l2_mean: 0.525468283481356
- pred_to_base_cosine_mean: 0.8497741905891377
- pred_to_base_mean_delta_mean: 0.0337054236678987
- pred_to_base_std_delta_mean: -0.06779391469730847
- pred_l1_improvement_vs_base_mean: 0.1209615353440893
- pred_relative_l2_improvement_vs_base_mean: 0.19789923754507216
- count: 552
- actual_payload_bpp_mean: 0.013999109682829483
- pred_condition_l1_win_rate: 0.9963768115942029
- pred_relative_l2_win_rate: 0.9963768115942029
- condition_residual_scale: 0.85
- condition_residual_tanh: True
- condition_delta_ablation: normal
- semantic_latent_ablation: normal
- detail_context_ablation: normal
- ablation_shuffle_seed: 1234
- payload_policy: analysis only; predicted condition is deterministic from decoded CoSER tensors, fixed adapter weights, and fixed CoD-Lite weights, so no image-specific side information is introduced

## Artifacts

- summary: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailcontrast_stats_spectrum_ft800_b8_condition_stats_full552/summary.json`
- per_image_condition_stats: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailcontrast_stats_spectrum_ft800_b8_condition_stats_full552/per_image_condition_stats.jsonl`
- channel_stats: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailcontrast_stats_spectrum_ft800_b8_condition_stats_full552/channel_stats.json`
- activation_histograms: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailcontrast_stats_spectrum_ft800_b8_condition_stats_full552/activation_histograms.json`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_154904-ezyx19dg`
