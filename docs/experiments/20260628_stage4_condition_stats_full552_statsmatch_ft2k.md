# 20260628_stage4_condition_stats_full552_statsmatch_ft2k

Date: 2026-06-28T21:59:58

## Command

```bash
scripts/analyze_stage4_cod_lite_condition_stats.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_statsmatch_ft2k_b8.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl --per-image-metrics results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/per_image_metrics.jsonl --run-name 20260628_stage4_condition_stats_full552_statsmatch_ft2k --batch-size 1 --num-workers 2 --wandb-mode offline
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
- pred_mean_mean: -0.014550142451750049
- pred_std_mean: 0.7282477075207061
- pred_min_mean: -9.156023550724637
- pred_max_mean: 8.491649682971014
- pred_abs_mean_mean: 0.5061051985923795
- pred_rms_mean: 0.7286461188957312
- pred_l2_mean: 456.91165935129357
- pred_channel_mean_std_mean: 0.4369949552675952
- pred_channel_std_mean_mean: 0.5072327131486457
- pred_channel_std_std_mean: 0.2622692015939865
- pred_spatial_highfreq_ratio_mean: 0.23488127272846043
- adapter_delta_raw_mean_mean: 0.08020586534605725
- adapter_delta_raw_std_mean: 0.95065195016239
- adapter_delta_raw_min_mean: -7.203181612318841
- adapter_delta_raw_max_mean: 9.377434329710145
- adapter_delta_raw_abs_mean_mean: 0.6618927205386369
- adapter_delta_raw_rms_mean: 0.9541477836344553
- adapter_delta_raw_l2_mean: 598.3168447190437
- adapter_delta_raw_channel_mean_std_mean: 0.773702347084232
- adapter_delta_raw_channel_std_mean_mean: 0.5023734063665936
- adapter_delta_raw_channel_std_std_mean: 0.20995714530294787
- adapter_delta_raw_spatial_highfreq_ratio_mean: 0.23097553970697132
- base_to_target_l1_mean: 0.5370939215780168
- base_to_target_mse_mean: 0.4842399032381566
- base_to_target_rms_mean: 0.6911686438581218
- base_to_target_relative_l2_mean: 0.8758228153422258
- base_to_target_cosine_mean: 0.6221415235836437
- base_to_target_mean_delta_mean: -0.03866021500832268
- base_to_target_std_delta_mean: 0.027971822282542354
- pred_to_target_l1_mean: 0.41646995249649754
- pred_to_target_mse_mean: 0.2911298201297936
- pred_to_target_rms_mean: 0.5371777859610923
- pred_to_target_relative_l2_mean: 0.6797684694005959
- pred_to_target_cosine_mean: 0.7462594938882883
- pred_to_target_mean_delta_mean: -0.006044465683653075
- pred_to_target_std_delta_mean: -0.06717005523218625
- pred_to_base_l1_mean: 0.3437089964855408
- pred_to_base_mse_mean: 0.16610174221189125
- pred_to_base_rms_mean: 0.4061533071208691
- pred_to_base_relative_l2_mean: 0.4966468287334926
- pred_to_base_cosine_mean: 0.8652287698958231
- pred_to_base_mean_delta_mean: 0.0326157494539233
- pred_to_base_std_delta_mean: -0.0951418775147286
- pred_l1_improvement_vs_base_mean: 0.1206239690815193
- pred_relative_l2_improvement_vs_base_mean: 0.19605434594162996
- count: 552
- actual_payload_bpp_mean: 0.013999109682829483
- pred_condition_l1_win_rate: 0.9945652173913043
- pred_relative_l2_win_rate: 0.9963768115942029
- condition_residual_scale: 0.75
- condition_residual_tanh: True
- payload_policy: analysis only; predicted condition is deterministic from decoded CoSER tensors, fixed adapter weights, and fixed CoD-Lite weights, so no image-specific side information is introduced

## Artifacts

- summary: `results/stage4_cod_lite_condition_stats/20260628_stage4_condition_stats_full552_statsmatch_ft2k/summary.json`
- per_image_condition_stats: `results/stage4_cod_lite_condition_stats/20260628_stage4_condition_stats_full552_statsmatch_ft2k/per_image_condition_stats.jsonl`
- channel_stats: `results/stage4_cod_lite_condition_stats/20260628_stage4_condition_stats_full552_statsmatch_ft2k/channel_stats.json`
- activation_histograms: `results/stage4_cod_lite_condition_stats/20260628_stage4_condition_stats_full552_statsmatch_ft2k/activation_histograms.json`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_215930-tsha3g15`
