# 20260629_stage4_detailtarget_ft700_b8_condition_stats_full552

Date: 2026-06-29T16:13:59

## Command

```bash
scripts/analyze_stage4_cod_lite_condition_stats.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailtarget_ft700_b8.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl --per-image-metrics results/stage4_cod_lite_adapter_eval/20260629_stage4_detailtarget_ft700_b8_full552_eval/per_image_metrics.jsonl --run-name 20260629_stage4_detailtarget_ft700_b8_condition_stats_full552 --crop-size 512 --batch-size 8 --num-workers 4 --wandb-mode offline
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
- pred_mean_mean: -0.013838319280170175
- pred_std_mean: 0.7384459415207738
- pred_min_mean: -9.042883831521738
- pred_max_mean: 8.437669836956522
- pred_abs_mean_mean: 0.5185318265812121
- pred_rms_mean: 0.7388351539122886
- pred_l2_mean: 463.3008912127951
- pred_channel_mean_std_mean: 0.43787776527629385
- pred_channel_std_mean_mean: 0.5247819481444531
- pred_channel_std_std_mean: 0.25516938534227834
- pred_spatial_highfreq_ratio_mean: 0.24150699131406736
- adapter_delta_raw_mean_mean: 0.06933664835894993
- adapter_delta_raw_std_mean: 0.8295038489867812
- adapter_delta_raw_min_mean: -6.860082653985507
- adapter_delta_raw_max_mean: 9.674620697463768
- adapter_delta_raw_abs_mean_mean: 0.5607808515213538
- adapter_delta_raw_rms_mean: 0.8325321800682856
- adapter_delta_raw_l2_mean: 522.055424897567
- adapter_delta_raw_channel_mean_std_mean: 0.6673452145271543
- adapter_delta_raw_channel_std_mean_mean: 0.4394977073306623
- adapter_delta_raw_channel_std_std_mean: 0.20558239592482214
- adapter_delta_raw_spatial_highfreq_ratio_mean: 0.23261640120999535
- base_to_target_l1_mean: 0.5373085212988266
- base_to_target_mse_mean: 0.48462129287097766
- base_to_target_rms_mean: 0.6914442249074362
- base_to_target_relative_l2_mean: 0.8762243978031303
- base_to_target_cosine_mean: 0.6218612093235487
- base_to_target_mean_delta_mean: -0.038658941766935524
- base_to_target_std_delta_mean: 0.028074710913326428
- pred_to_target_l1_mean: 0.4095985118461692
- pred_to_target_mse_mean: 0.2811517246013534
- pred_to_target_rms_mean: 0.5278148027001948
- pred_to_target_relative_l2_mean: 0.6680526887809021
- pred_to_target_cosine_mean: 0.7574867041430612
- pred_to_target_mean_delta_mean: -0.005356300933759696
- pred_to_target_std_delta_mean: -0.056945839761823845
- pred_to_base_l1_mean: 0.34765504462563473
- pred_to_base_mse_mean: 0.17651745870007554
- pred_to_base_rms_mean: 0.4181247764415499
- pred_to_base_relative_l2_mean: 0.5112832727423613
- pred_to_base_cosine_mean: 0.8568114225847133
- pred_to_base_mean_delta_mean: 0.033302640733882734
- pred_to_base_std_delta_mean: -0.08502055067515028
- pred_l1_improvement_vs_base_mean: 0.12771000945265742
- pred_relative_l2_improvement_vs_base_mean: 0.20817170902222826
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

- summary: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailtarget_ft700_b8_condition_stats_full552/summary.json`
- per_image_condition_stats: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailtarget_ft700_b8_condition_stats_full552/per_image_condition_stats.jsonl`
- channel_stats: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailtarget_ft700_b8_condition_stats_full552/channel_stats.json`
- activation_histograms: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailtarget_ft700_b8_condition_stats_full552/activation_histograms.json`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_161339-di581xqh`
