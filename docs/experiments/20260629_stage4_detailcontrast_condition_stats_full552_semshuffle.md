# 20260629_stage4_detailcontrast_condition_stats_full552_semshuffle

Date: 2026-06-29T15:34:46

## Command

```bash
scripts/analyze_stage4_cod_lite_condition_stats.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl --per-image-metrics results/stage4_cod_lite_adapter_eval/20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8_full552_eval/per_image_metrics.jsonl --run-name 20260629_stage4_detailcontrast_condition_stats_full552_semshuffle --crop-size 512 --batch-size 8 --num-workers 4 --semantic-latent-ablation shuffle --ablation-shuffle-seed 1234 --wandb-mode offline
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
- pred_mean_mean: -0.014748659892038104
- pred_std_mean: 0.7177982548440712
- pred_min_mean: -9.017493206521738
- pred_max_mean: 8.365319293478262
- pred_abs_mean_mean: 0.502658548940351
- pred_rms_mean: 0.7181960246053295
- pred_l2_mean: 450.3587248429008
- pred_channel_mean_std_mean: 0.4250140275968158
- pred_channel_std_mean_mean: 0.5115585681320964
- pred_channel_std_std_mean: 0.24736802107182101
- pred_spatial_highfreq_ratio_mean: 0.24464399672612763
- adapter_delta_raw_mean_mean: 0.07244902861682509
- adapter_delta_raw_std_mean: 0.8828121748739395
- adapter_delta_raw_min_mean: -7.290619338768116
- adapter_delta_raw_max_mean: 10.677423007246377
- adapter_delta_raw_abs_mean_mean: 0.5882520564440368
- adapter_delta_raw_rms_mean: 0.8859107237903104
- adapter_delta_raw_l2_mean: 555.5274747212728
- adapter_delta_raw_channel_mean_std_mean: 0.7170416421756365
- adapter_delta_raw_channel_std_mean_mean: 0.4546186507180117
- adapter_delta_raw_channel_std_std_mean: 0.22565909684298263
- adapter_delta_raw_spatial_highfreq_ratio_mean: 0.21973246551942135
- base_to_target_l1_mean: 0.5373085212988266
- base_to_target_mse_mean: 0.48462129287097766
- base_to_target_rms_mean: 0.6914442249074362
- base_to_target_relative_l2_mean: 0.8762243978031303
- base_to_target_cosine_mean: 0.6218612093235487
- base_to_target_mean_delta_mean: -0.038658941766935524
- base_to_target_std_delta_mean: 0.028074710913326428
- pred_to_target_l1_mean: 0.4148419083136579
- pred_to_target_mse_mean: 0.28737933763667295
- pred_to_target_rms_mean: 0.533915098702562
- pred_to_target_relative_l2_mean: 0.6755938447903895
- pred_to_target_cosine_mean: 0.7477938807097034
- pred_to_target_mean_delta_mean: -0.0062666415456671684
- pred_to_target_std_delta_mean: -0.07759352643852649
- pred_to_base_l1_mean: 0.35655538138488063
- pred_to_base_mse_mean: 0.18464310475341653
- pred_to_base_rms_mean: 0.42773518722126447
- pred_to_base_relative_l2_mean: 0.5230975807037043
- pred_to_base_cosine_mean: 0.8491352502850519
- pred_to_base_mean_delta_mean: 0.03239230005124557
- pred_to_base_std_delta_mean: -0.10566823735185292
- pred_l1_improvement_vs_base_mean: 0.12246661298516867
- pred_relative_l2_improvement_vs_base_mean: 0.2006305530127408
- count: 552
- actual_payload_bpp_mean: 0.013999109682829483
- pred_condition_l1_win_rate: 0.9945652173913043
- pred_relative_l2_win_rate: 0.9963768115942029
- condition_residual_scale: 0.85
- condition_residual_tanh: True
- condition_delta_ablation: normal
- semantic_latent_ablation: shuffle
- detail_context_ablation: normal
- ablation_shuffle_seed: 1234
- payload_policy: analysis only; predicted condition is deterministic from decoded CoSER tensors, fixed adapter weights, and fixed CoD-Lite weights, so no image-specific side information is introduced

## Artifacts

- summary: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailcontrast_condition_stats_full552_semshuffle/summary.json`
- per_image_condition_stats: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailcontrast_condition_stats_full552_semshuffle/per_image_condition_stats.jsonl`
- channel_stats: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailcontrast_condition_stats_full552_semshuffle/channel_stats.json`
- activation_histograms: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailcontrast_condition_stats_full552_semshuffle/activation_histograms.json`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_153426-g5ubiul4`
