# 20260629_stage4_detailcontrast_condition_stats_full552_semzero

Date: 2026-06-29T15:34:03

## Command

```bash
scripts/analyze_stage4_cod_lite_condition_stats.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl --per-image-metrics results/stage4_cod_lite_adapter_eval/20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8_full552_eval/per_image_metrics.jsonl --run-name 20260629_stage4_detailcontrast_condition_stats_full552_semzero --crop-size 512 --batch-size 8 --num-workers 4 --semantic-latent-ablation zero --wandb-mode offline
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
- pred_mean_mean: -0.014062116475681409
- pred_std_mean: 0.7314483374357224
- pred_min_mean: -9.03031589673913
- pred_max_mean: 8.39068161231884
- pred_abs_mean_mean: 0.513810840745767
- pred_rms_mean: 0.7318328590928644
- pred_l2_mean: 458.9099653492803
- pred_channel_mean_std_mean: 0.4324041157960892
- pred_channel_std_mean_mean: 0.5214901841313079
- pred_channel_std_std_mean: 0.25181496510471124
- pred_spatial_highfreq_ratio_mean: 0.24262803849642692
- adapter_delta_raw_mean_mean: 0.07232937701316415
- adapter_delta_raw_std_mean: 0.8700380802586458
- adapter_delta_raw_min_mean: -7.015200407608695
- adapter_delta_raw_max_mean: 10.29387454710145
- adapter_delta_raw_abs_mean_mean: 0.5808767015519349
- adapter_delta_raw_rms_mean: 0.8731777649102868
- adapter_delta_raw_l2_mean: 547.5430271314538
- adapter_delta_raw_channel_mean_std_mean: 0.707794215368188
- adapter_delta_raw_channel_std_mean_mean: 0.4465920879789021
- adapter_delta_raw_channel_std_std_mean: 0.21803202377497286
- adapter_delta_raw_spatial_highfreq_ratio_mean: 0.22037620392992444
- base_to_target_l1_mean: 0.5373085212988266
- base_to_target_mse_mean: 0.48462129287097766
- base_to_target_rms_mean: 0.6914442249074362
- base_to_target_relative_l2_mean: 0.8762243978031303
- base_to_target_cosine_mean: 0.6218612093235487
- base_to_target_mean_delta_mean: -0.038658941766935524
- base_to_target_std_delta_mean: 0.028074710913326428
- pred_to_target_l1_mean: 0.41196326339158457
- pred_to_target_mse_mean: 0.28396785299739113
- pred_to_target_rms_mean: 0.5305575394760007
- pred_to_target_relative_l2_mean: 0.6714039672220099
- pred_to_target_cosine_mean: 0.7537542946528697
- pred_to_target_mean_delta_mean: -0.005580098172415909
- pred_to_target_std_delta_mean: -0.06394344384687534
- pred_to_base_l1_mean: 0.35270700528137927
- pred_to_base_mse_mean: 0.181689176964911
- pred_to_base_rms_mean: 0.4238479740485765
- pred_to_base_relative_l2_mean: 0.5186062238138655
- pred_to_base_cosine_mean: 0.8518090548290722
- pred_to_base_mean_delta_mean: 0.03307884353621309
- pred_to_base_std_delta_mean: -0.09201815476020177
- pred_l1_improvement_vs_base_mean: 0.12534525790724202
- pred_relative_l2_improvement_vs_base_mean: 0.20482043058112048
- count: 552
- actual_payload_bpp_mean: 0.013999109682829483
- pred_condition_l1_win_rate: 0.9963768115942029
- pred_relative_l2_win_rate: 0.9963768115942029
- condition_residual_scale: 0.85
- condition_residual_tanh: True
- condition_delta_ablation: normal
- semantic_latent_ablation: zero
- detail_context_ablation: normal
- ablation_shuffle_seed: 1234
- payload_policy: analysis only; predicted condition is deterministic from decoded CoSER tensors, fixed adapter weights, and fixed CoD-Lite weights, so no image-specific side information is introduced

## Artifacts

- summary: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailcontrast_condition_stats_full552_semzero/summary.json`
- per_image_condition_stats: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailcontrast_condition_stats_full552_semzero/per_image_condition_stats.jsonl`
- channel_stats: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailcontrast_condition_stats_full552_semzero/channel_stats.json`
- activation_histograms: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailcontrast_condition_stats_full552_semzero/activation_histograms.json`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_153343-6u2dmb2i`
