# 20260629_stage4_detailtarget_ft700_b8_condition_stats_full552_detailzero

Date: 2026-06-29T16:14:35

## Command

```bash
scripts/analyze_stage4_cod_lite_condition_stats.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailtarget_ft700_b8.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl --per-image-metrics results/stage4_cod_lite_adapter_eval/20260629_stage4_detailtarget_ft700_b8_full552_eval/per_image_metrics.jsonl --run-name 20260629_stage4_detailtarget_ft700_b8_condition_stats_full552_detailzero --crop-size 512 --batch-size 8 --num-workers 4 --detail-context-ablation zero --wandb-mode offline
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
- pred_mean_mean: -0.013622927551019213
- pred_std_mean: 0.7373549089483593
- pred_min_mean: -9.041808197463768
- pred_max_mean: 8.433848505434783
- pred_abs_mean_mean: 0.5176559850249601
- pred_rms_mean: 0.737742265821367
- pred_l2_mean: 462.61557415948397
- pred_channel_mean_std_mean: 0.4395585896960203
- pred_channel_std_mean_mean: 0.5223572854546533
- pred_channel_std_std_mean: 0.25431324795320415
- pred_spatial_highfreq_ratio_mean: 0.24291803574432497
- adapter_delta_raw_mean_mean: 0.06719464275216602
- adapter_delta_raw_std_mean: 0.7848652904124364
- adapter_delta_raw_min_mean: -6.598703577898551
- adapter_delta_raw_max_mean: 8.80364017210145
- adapter_delta_raw_abs_mean_mean: 0.5341697725588861
- adapter_delta_raw_rms_mean: 0.7878688962563224
- adapter_delta_raw_l2_mean: 494.04844853497934
- adapter_delta_raw_channel_mean_std_mean: 0.6260446490368982
- adapter_delta_raw_channel_std_mean_mean: 0.4255825821040333
- adapter_delta_raw_channel_std_std_mean: 0.19261770630660263
- adapter_delta_raw_spatial_highfreq_ratio_mean: 0.2368206651746363
- base_to_target_l1_mean: 0.5373085212988266
- base_to_target_mse_mean: 0.48462129287097766
- base_to_target_rms_mean: 0.6914442249074362
- base_to_target_relative_l2_mean: 0.8762243978031303
- base_to_target_cosine_mean: 0.6218612093235487
- base_to_target_mean_delta_mean: -0.038658941766935524
- base_to_target_std_delta_mean: 0.028074710913326428
- pred_to_target_l1_mean: 0.41130215377695317
- pred_to_target_mse_mean: 0.2836213187193093
- pred_to_target_rms_mean: 0.5299728914447452
- pred_to_target_relative_l2_mean: 0.6707911437404328
- pred_to_target_cosine_mean: 0.7550614199776581
- pred_to_target_mean_delta_mean: -0.00514090928460802
- pred_to_target_std_delta_mean: -0.05803687233423841
- pred_to_base_l1_mean: 0.33858515333006345
- pred_to_base_mse_mean: 0.16839471833267505
- pred_to_base_rms_mean: 0.408665517318076
- pred_to_base_relative_l2_mean: 0.4995452991646269
- pred_to_base_cosine_mean: 0.8638245346753494
- pred_to_base_mean_delta_mean: 0.033518032391852554
- pred_to_base_std_delta_mean: -0.08611158324756484
- pred_l1_improvement_vs_base_mean: 0.1260063675218734
- pred_relative_l2_improvement_vs_base_mean: 0.2054332540626975
- count: 552
- actual_payload_bpp_mean: 0.013999109682829483
- pred_condition_l1_win_rate: 1.0
- pred_relative_l2_win_rate: 1.0
- condition_residual_scale: 0.85
- condition_residual_tanh: True
- condition_delta_ablation: normal
- semantic_latent_ablation: normal
- detail_context_ablation: zero
- ablation_shuffle_seed: 1234
- payload_policy: analysis only; predicted condition is deterministic from decoded CoSER tensors, fixed adapter weights, and fixed CoD-Lite weights, so no image-specific side information is introduced

## Artifacts

- summary: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailtarget_ft700_b8_condition_stats_full552_detailzero/summary.json`
- per_image_condition_stats: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailtarget_ft700_b8_condition_stats_full552_detailzero/per_image_condition_stats.jsonl`
- channel_stats: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailtarget_ft700_b8_condition_stats_full552_detailzero/channel_stats.json`
- activation_histograms: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailtarget_ft700_b8_condition_stats_full552_detailzero/activation_histograms.json`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_161414-n5bun4p7`
