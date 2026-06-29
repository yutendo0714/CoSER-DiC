# 20260629_stage4_detailtarget_perceptual_ft700_b8_condition_stats_full552_detailshuffle

Date: 2026-06-29T16:29:46

## Command

```bash
scripts/analyze_stage4_cod_lite_condition_stats.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailtarget_perceptual_ft700_b8.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl --per-image-metrics results/stage4_cod_lite_adapter_eval/20260629_stage4_detailtarget_perceptual_ft700_b8_full552_eval/per_image_metrics.jsonl --run-name 20260629_stage4_detailtarget_perceptual_ft700_b8_condition_stats_full552_detailshuffle --crop-size 512 --batch-size 8 --num-workers 4 --detail-context-ablation shuffle --wandb-mode offline
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
- pred_mean_mean: -0.014130399221008649
- pred_std_mean: 0.7360729053616524
- pred_min_mean: -9.041610054347826
- pred_max_mean: 8.428979846014492
- pred_abs_mean_mean: 0.5165504015766192
- pred_rms_mean: 0.7364736607541209
- pred_l2_mean: 461.82007178707397
- pred_channel_mean_std_mean: 0.43689957055924594
- pred_channel_std_mean_mean: 0.5229340021808943
- pred_channel_std_std_mean: 0.2542713828033943
- pred_spatial_highfreq_ratio_mean: 0.2443992649137542
- adapter_delta_raw_mean_mean: 0.06946656956890787
- adapter_delta_raw_std_mean: 0.8323394066073756
- adapter_delta_raw_min_mean: -7.004132699275362
- adapter_delta_raw_max_mean: 9.605610280797102
- adapter_delta_raw_abs_mean_mean: 0.5616164096239684
- adapter_delta_raw_rms_mean: 0.8353642682018487
- adapter_delta_raw_l2_mean: 523.8313411215078
- adapter_delta_raw_channel_mean_std_mean: 0.6679316118251586
- adapter_delta_raw_channel_std_mean_mean: 0.44372594351137895
- adapter_delta_raw_channel_std_std_mean: 0.20748384056639843
- adapter_delta_raw_spatial_highfreq_ratio_mean: 0.2303215786949664
- base_to_target_l1_mean: 0.5373085212988266
- base_to_target_mse_mean: 0.48462129287097766
- base_to_target_rms_mean: 0.6914442249074362
- base_to_target_relative_l2_mean: 0.8762243978031303
- base_to_target_cosine_mean: 0.6218612093235487
- base_to_target_mean_delta_mean: -0.038658941766935524
- base_to_target_std_delta_mean: 0.028074710913326428
- pred_to_target_l1_mean: 0.4118484816905381
- pred_to_target_mse_mean: 0.28422929926950863
- pred_to_target_rms_mean: 0.5305989097425903
- pred_to_target_relative_l2_mean: 0.6715344038886436
- pred_to_target_cosine_mean: 0.7543381352679454
- pred_to_target_mean_delta_mean: -0.005648381062482455
- pred_to_target_std_delta_mean: -0.05931887592094532
- pred_to_base_l1_mean: 0.3483140717489996
- pred_to_base_mse_mean: 0.17697411455699932
- pred_to_base_rms_mean: 0.41883701998470485
- pred_to_base_relative_l2_mean: 0.5120977527205495
- pred_to_base_cosine_mean: 0.8564024393757185
- pred_to_base_mean_delta_mean: 0.033010560624382415
- pred_to_base_std_delta_mean: -0.08739358683427174
- pred_l1_improvement_vs_base_mean: 0.1254600396082885
- pred_relative_l2_improvement_vs_base_mean: 0.2046899939144867
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

- summary: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailtarget_perceptual_ft700_b8_condition_stats_full552_detailshuffle/summary.json`
- per_image_condition_stats: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailtarget_perceptual_ft700_b8_condition_stats_full552_detailshuffle/per_image_condition_stats.jsonl`
- channel_stats: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailtarget_perceptual_ft700_b8_condition_stats_full552_detailshuffle/channel_stats.json`
- activation_histograms: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailtarget_perceptual_ft700_b8_condition_stats_full552_detailshuffle/activation_histograms.json`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_162926-aqp1swqv`
