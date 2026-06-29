# 20260629_stage4_detailcontrast_condition_stats_full552_detailshuffle

Date: 2026-06-29T15:36:02

## Command

```bash
scripts/analyze_stage4_cod_lite_condition_stats.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl --per-image-metrics results/stage4_cod_lite_adapter_eval/20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8_full552_eval/per_image_metrics.jsonl --run-name 20260629_stage4_detailcontrast_condition_stats_full552_detailshuffle --crop-size 512 --batch-size 8 --num-workers 4 --detail-context-ablation shuffle --ablation-shuffle-seed 1234 --wandb-mode offline
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
- pred_mean_mean: -0.013854450131625395
- pred_std_mean: 0.7399589196927305
- pred_min_mean: -9.045516304347826
- pred_max_mean: 8.436169610507246
- pred_abs_mean_mean: 0.520082694194887
- pred_rms_mean: 0.7403406571002974
- pred_l2_mean: 464.24494641069055
- pred_channel_mean_std_mean: 0.4361661282246527
- pred_channel_std_mean_mean: 0.5283383763786675
- pred_channel_std_std_mean: 0.25493156761470914
- pred_spatial_highfreq_ratio_mean: 0.24096447375157606
- adapter_delta_raw_mean_mean: 0.07121149078011513
- adapter_delta_raw_std_mean: 0.849640044235233
- adapter_delta_raw_min_mean: -6.961163949275362
- adapter_delta_raw_max_mean: 9.934867527173912
- adapter_delta_raw_abs_mean_mean: 0.569938966329547
- adapter_delta_raw_rms_mean: 0.8527547899579656
- adapter_delta_raw_l2_mean: 534.7364046953727
- adapter_delta_raw_channel_mean_std_mean: 0.6878068860134353
- adapter_delta_raw_channel_std_mean_mean: 0.4431062687890253
- adapter_delta_raw_channel_std_std_mean: 0.21066381006191173
- adapter_delta_raw_spatial_highfreq_ratio_mean: 0.22702183605482182
- base_to_target_l1_mean: 0.5373085212988266
- base_to_target_mse_mean: 0.48462129287097766
- base_to_target_rms_mean: 0.6914442249074362
- base_to_target_relative_l2_mean: 0.8762243978031303
- base_to_target_cosine_mean: 0.6218612093235487
- base_to_target_mean_delta_mean: -0.038658941766935524
- base_to_target_std_delta_mean: 0.028074710913326428
- pred_to_target_l1_mean: 0.4115995007256667
- pred_to_target_mse_mean: 0.2836842462690412
- pred_to_target_rms_mean: 0.5301962986696458
- pred_to_target_relative_l2_mean: 0.670989072787157
- pred_to_target_cosine_mean: 0.7558142385189084
- pred_to_target_mean_delta_mean: -0.00537243182333955
- pred_to_target_std_delta_mean: -0.055432861589867134
- pred_to_base_l1_mean: 0.34980357681279595
- pred_to_base_mse_mean: 0.1787519045714019
- pred_to_base_rms_mean: 0.42066172672354657
- pred_to_base_relative_l2_mean: 0.5145079717040062
- pred_to_base_cosine_mean: 0.8549064620249514
- pred_to_base_mean_delta_mean: 0.033286509900421334
- pred_to_base_std_delta_mean: -0.08350757250319356
- pred_l1_improvement_vs_base_mean: 0.1257090205731599
- pred_relative_l2_improvement_vs_base_mean: 0.20523532501597336
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

- summary: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailcontrast_condition_stats_full552_detailshuffle/summary.json`
- per_image_condition_stats: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailcontrast_condition_stats_full552_detailshuffle/per_image_condition_stats.jsonl`
- channel_stats: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailcontrast_condition_stats_full552_detailshuffle/channel_stats.json`
- activation_histograms: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailcontrast_condition_stats_full552_detailshuffle/activation_histograms.json`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_153541-ps4rtwop`
