# 20260628_stage4_condition_stats_full552_ablate_semzero

Date: 2026-06-28T23:30:10

## Command

```bash
scripts/analyze_stage4_cod_lite_condition_stats.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_statsmatch_ft2k_b8.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl --per-image-metrics results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/per_image_metrics.jsonl --run-name 20260628_stage4_condition_stats_full552_ablate_semzero --crop-size 512 --batch-size 1 --num-workers 2 --semantic-latent-ablation zero --wandb-mode offline
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
- pred_mean_mean: -0.012321679951352395
- pred_std_mean: 0.6970992360426032
- pred_min_mean: -9.123301630434783
- pred_max_mean: 8.415930706521738
- pred_abs_mean_mean: 0.4840303488291692
- pred_rms_mean: 0.6974625977269118
- pred_l2_mean: 437.3574275141177
- pred_channel_mean_std_mean: 0.43704619388217514
- pred_channel_std_mean_mean: 0.4740783409148023
- pred_channel_std_std_mean: 0.24292185111646203
- pred_spatial_highfreq_ratio_mean: 0.24678947544400243
- adapter_delta_raw_mean_mean: 0.07806127578015035
- adapter_delta_raw_std_mean: 1.1025736942023472
- adapter_delta_raw_min_mean: -13.651154891304348
- adapter_delta_raw_max_mean: 13.191576086956522
- adapter_delta_raw_abs_mean_mean: 0.7068237929024558
- adapter_delta_raw_rms_mean: 1.1061875364486722
- adapter_delta_raw_l2_mean: 693.6563173376995
- adapter_delta_raw_channel_mean_std_mean: 0.8065359911516957
- adapter_delta_raw_channel_std_mean_mean: 0.6080566942691803
- adapter_delta_raw_channel_std_std_mean: 0.3339206026192161
- adapter_delta_raw_spatial_highfreq_ratio_mean: 0.22273748548215497
- base_to_target_l1_mean: 0.5370939215780168
- base_to_target_mse_mean: 0.4842399032381566
- base_to_target_rms_mean: 0.6911686438581218
- base_to_target_relative_l2_mean: 0.8758228153422258
- base_to_target_cosine_mean: 0.6221415235836437
- base_to_target_mean_delta_mean: -0.03866021500832268
- base_to_target_std_delta_mean: 0.027971822282542354
- pred_to_target_l1_mean: 0.4298318863670895
- pred_to_target_mse_mean: 0.30934234260432963
- pred_to_target_rms_mean: 0.5536434204466101
- pred_to_target_relative_l2_mean: 0.7003887831300929
- pred_to_target_cosine_mean: 0.7240811446546644
- pred_to_target_mean_delta_mean: -0.0038160030714501827
- pred_to_target_std_delta_mean: -0.09831852671028911
- pred_to_base_l1_mean: 0.35013931330995285
- pred_to_base_mse_mean: 0.1711286635313561
- pred_to_base_rms_mean: 0.4123986957297809
- pred_to_base_relative_l2_mean: 0.5041458717938783
- pred_to_base_cosine_mean: 0.861452182026013
- pred_to_base_mean_delta_mean: 0.034844212104107486
- pred_to_base_std_delta_mean: -0.12629034899283145
- pred_l1_improvement_vs_base_mean: 0.10726203521092732
- pred_relative_l2_improvement_vs_base_mean: 0.175434032212133
- count: 552
- actual_payload_bpp_mean: 0.013999109682829483
- pred_condition_l1_win_rate: 0.9836956521739131
- pred_relative_l2_win_rate: 0.9855072463768116
- condition_residual_scale: 0.75
- condition_residual_tanh: True
- condition_delta_ablation: normal
- semantic_latent_ablation: zero
- detail_context_ablation: normal
- ablation_shuffle_seed: 1234
- payload_policy: analysis only; predicted condition is deterministic from decoded CoSER tensors, fixed adapter weights, and fixed CoD-Lite weights, so no image-specific side information is introduced

## Artifacts

- summary: `results/stage4_cod_lite_condition_stats/20260628_stage4_condition_stats_full552_ablate_semzero/summary.json`
- per_image_condition_stats: `results/stage4_cod_lite_condition_stats/20260628_stage4_condition_stats_full552_ablate_semzero/per_image_condition_stats.jsonl`
- channel_stats: `results/stage4_cod_lite_condition_stats/20260628_stage4_condition_stats_full552_ablate_semzero/channel_stats.json`
- activation_histograms: `results/stage4_cod_lite_condition_stats/20260628_stage4_condition_stats_full552_ablate_semzero/activation_histograms.json`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_232944-4iw8ub90`
