# 20260629_stage4_detailcontrast_condition_stats_full552_detailzero

Date: 2026-06-29T15:35:22

## Command

```bash
scripts/analyze_stage4_cod_lite_condition_stats.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl --per-image-metrics results/stage4_cod_lite_adapter_eval/20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8_full552_eval/per_image_metrics.jsonl --run-name 20260629_stage4_detailcontrast_condition_stats_full552_detailzero --crop-size 512 --batch-size 8 --num-workers 4 --detail-context-ablation zero --wandb-mode offline
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
- pred_mean_mean: -0.013633330253825356
- pred_std_mean: 0.740145122119482
- pred_min_mean: -9.045431385869565
- pred_max_mean: 8.434273097826088
- pred_abs_mean_mean: 0.520321968672932
- pred_rms_mean: 0.7405217897848807
- pred_l2_mean: 464.35852780549425
- pred_channel_mean_std_mean: 0.43728226142517035
- pred_channel_std_mean_mean: 0.5277691177375938
- pred_channel_std_std_mean: 0.25472643131903117
- pred_spatial_highfreq_ratio_mean: 0.24125445563940034
- adapter_delta_raw_mean_mean: 0.0706117616939372
- adapter_delta_raw_std_mean: 0.8336339507090009
- adapter_delta_raw_min_mean: -6.876471920289855
- adapter_delta_raw_max_mean: 9.681244338768115
- adapter_delta_raw_abs_mean_mean: 0.5603160459710204
- adapter_delta_raw_rms_mean: 0.8367581845841546
- adapter_delta_raw_l2_mean: 524.7054228851761
- adapter_delta_raw_channel_mean_std_mean: 0.6731225975073766
- adapter_delta_raw_channel_std_mean_mean: 0.43794897884346434
- adapter_delta_raw_channel_std_std_mean: 0.20599321829344053
- adapter_delta_raw_spatial_highfreq_ratio_mean: 0.22932318280842426
- base_to_target_l1_mean: 0.5373085212988266
- base_to_target_mse_mean: 0.48462129287097766
- base_to_target_rms_mean: 0.6914442249074362
- base_to_target_relative_l2_mean: 0.8762243978031303
- base_to_target_cosine_mean: 0.6218612093235487
- base_to_target_mean_delta_mean: -0.038658941766935524
- base_to_target_std_delta_mean: 0.028074710913326428
- pred_to_target_l1_mean: 0.41133324235029844
- pred_to_target_mse_mean: 0.2833825385710899
- pred_to_target_rms_mean: 0.5298825042403262
- pred_to_target_relative_l2_mean: 0.6705931924499463
- pred_to_target_cosine_mean: 0.7561316157596699
- pred_to_target_mean_delta_mean: -0.005151312051141403
- pred_to_target_std_delta_mean: -0.055246659163115684
- pred_to_base_l1_mean: 0.3465571381637584
- pred_to_base_mse_mean: 0.1758355290858426
- pred_to_base_rms_mean: 0.4173070800801118
- pred_to_base_relative_l2_mean: 0.5103532596342806
- pred_to_base_cosine_mean: 0.8574426588804825
- pred_to_base_mean_delta_mean: 0.033507629540627415
- pred_to_base_std_delta_mean: -0.08332137007644211
- pred_l1_improvement_vs_base_mean: 0.12597527894852817
- pred_relative_l2_improvement_vs_base_mean: 0.20563120535318402
- count: 552
- actual_payload_bpp_mean: 0.013999109682829483
- pred_condition_l1_win_rate: 0.9981884057971014
- pred_relative_l2_win_rate: 1.0
- condition_residual_scale: 0.85
- condition_residual_tanh: True
- condition_delta_ablation: normal
- semantic_latent_ablation: normal
- detail_context_ablation: zero
- ablation_shuffle_seed: 1234
- payload_policy: analysis only; predicted condition is deterministic from decoded CoSER tensors, fixed adapter weights, and fixed CoD-Lite weights, so no image-specific side information is introduced

## Artifacts

- summary: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailcontrast_condition_stats_full552_detailzero/summary.json`
- per_image_condition_stats: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailcontrast_condition_stats_full552_detailzero/per_image_condition_stats.jsonl`
- channel_stats: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailcontrast_condition_stats_full552_detailzero/channel_stats.json`
- activation_histograms: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailcontrast_condition_stats_full552_detailzero/activation_histograms.json`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_153501-klzzgbhh`
