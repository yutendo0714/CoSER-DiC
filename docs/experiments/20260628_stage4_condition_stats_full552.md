# 20260628_stage4_condition_stats_full552

Date: 2026-06-28T21:49:11

## Command

```bash
scripts/analyze_stage4_cod_lite_condition_stats.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_tanh075_traincache2048_5k_b8.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl --per-image-metrics results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/per_image_metrics.jsonl --run-name 20260628_stage4_condition_stats_full552 --batch-size 1 --num-workers 2 --wandb-mode offline
```

## Summary

- target_mean_mean: -0.008505676841502424
- target_std_mean: 0.7954177627528923
- target_min_mean: -8.459210824275363
- target_max_mean: 8.118602807971014
- target_abs_mean_mean: 0.5763484449058339
- target_rms_mean: 0.7957758316095325
- target_l2_mean: 499.00664597663325
- target_channel_mean_std_mean: 0.4745923965305522
- target_channel_std_mean_mean: 0.575071699053481
- target_channel_std_std_mean: 0.24980104682238205
- target_spatial_highfreq_ratio_mean: 0.25754632082754286
- base_mean_mean: -0.047165891781122
- base_std_mean: 0.8233895850354347
- base_min_mean: -9.864809782608695
- base_max_mean: 9.09762794384058
- base_abs_mean_mean: 0.5892797345700471
- base_rms_mean: 0.8250267163350962
- base_l2_mean: 517.3489790100982
- base_channel_mean_std_mean: 0.5554191807581894
- base_channel_std_mean_mean: 0.535575217526892
- base_channel_std_std_mean: 0.2661126420792678
- base_spatial_highfreq_ratio_mean: 0.2776259911934967
- pred_mean_mean: -0.015140138125468144
- pred_std_mean: 0.6932058052524276
- pred_min_mean: -9.134171195652174
- pred_max_mean: 8.44732223731884
- pred_abs_mean_mean: 0.4736300892588021
- pred_rms_mean: 0.6936390063037043
- pred_l2_mean: 434.95977136363155
- pred_channel_mean_std_mean: 0.42806959308791853
- pred_channel_std_mean_mean: 0.4648459815028785
- pred_channel_std_std_mean: 0.26201989696077677
- pred_spatial_highfreq_ratio_mean: 0.2279549147663773
- adapter_delta_raw_mean_mean: 0.07347476666968694
- adapter_delta_raw_std_mean: 0.8304711155917334
- adapter_delta_raw_min_mean: -5.927847599637682
- adapter_delta_raw_max_mean: 8.23528079710145
- adapter_delta_raw_abs_mean_mean: 0.5886345268159673
- adapter_delta_raw_rms_mean: 0.8338100057149279
- adapter_delta_raw_l2_mean: 522.856710627459
- adapter_delta_raw_channel_mean_std_mean: 0.6688753556514132
- adapter_delta_raw_channel_std_mean_mean: 0.453743241187455
- adapter_delta_raw_channel_std_std_mean: 0.17611333627955636
- adapter_delta_raw_spatial_highfreq_ratio_mean: 0.2983719664963259
- base_to_target_l1_mean: 0.5370939215780168
- base_to_target_mse_mean: 0.4842399032381566
- base_to_target_rms_mean: 0.6911686438581218
- base_to_target_relative_l2_mean: 0.8758228153422258
- base_to_target_cosine_mean: 0.6221415235836437
- base_to_target_mean_delta_mean: -0.03866021500832268
- base_to_target_std_delta_mean: 0.027971822282542354
- pred_to_target_l1_mean: 0.41117009050820186
- pred_to_target_mse_mean: 0.2842813791518194
- pred_to_target_rms_mean: 0.5307232030815836
- pred_to_target_relative_l2_mean: 0.6716151401076628
- pred_to_target_cosine_mean: 0.7447236403283002
- pred_to_target_mean_delta_mean: -0.0066344612453089
- pred_to_target_std_delta_mean: -0.10221195750046468
- pred_to_base_l1_mean: 0.3237944392460412
- pred_to_base_mse_mean: 0.1493260925905644
- pred_to_base_rms_mean: 0.38521942058983055
- pred_to_base_relative_l2_mean: 0.47114033948468126
- pred_to_base_cosine_mean: 0.8805094572944917
- pred_to_base_mean_delta_mean: 0.0320257537667572
- pred_to_base_std_delta_mean: -0.13018377978300702
- pred_l1_improvement_vs_base_mean: 0.12592383106981497
- pred_relative_l2_improvement_vs_base_mean: 0.2042076752345631
- count: 552
- actual_payload_bpp_mean: 0.013999109682829483
- pred_condition_l1_win_rate: 0.9981884057971014
- pred_relative_l2_win_rate: 0.9981884057971014
- condition_residual_scale: 0.75
- condition_residual_tanh: True
- payload_policy: analysis only; predicted condition is deterministic from decoded CoSER tensors, fixed adapter weights, and fixed CoD-Lite weights, so no image-specific side information is introduced

## Artifacts

- summary: `results/stage4_cod_lite_condition_stats/20260628_stage4_condition_stats_full552/summary.json`
- per_image_condition_stats: `results/stage4_cod_lite_condition_stats/20260628_stage4_condition_stats_full552/per_image_condition_stats.jsonl`
- channel_stats: `results/stage4_cod_lite_condition_stats/20260628_stage4_condition_stats_full552/channel_stats.json`
- activation_histograms: `results/stage4_cod_lite_condition_stats/20260628_stage4_condition_stats_full552/activation_histograms.json`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_214845-17vhtxlt`
