# 20260629_stage4_detailcontrast_condition_stats_full552

Date: 2026-06-29T15:30:20

## Command

```bash
scripts/analyze_stage4_cod_lite_condition_stats.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl --per-image-metrics results/stage4_cod_lite_adapter_eval/20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8_full552_eval/per_image_metrics.jsonl --run-name 20260629_stage4_detailcontrast_condition_stats_full552 --crop-size 512 --batch-size 4 --num-workers 4 --wandb-mode offline
```

## Summary

- target_mean_mean: -0.008519529968446253
- target_std_mean: 0.7953403763796972
- target_min_mean: -8.45513473731884
- target_max_mean: 8.126726675724637
- target_abs_mean_mean: 0.5762931728276653
- target_rms_mean: 0.7956981543397558
- target_l2_mean: 498.9579366601032
- target_channel_mean_std_mean: 0.474444260145875
- target_channel_std_mean_mean: 0.5750916341076726
- target_channel_std_std_mean: 0.24980937325112199
- target_spatial_highfreq_ratio_mean: 0.2575170497369507
- base_mean_mean: -0.04716410703553845
- base_std_mean: 0.823466194604618
- base_min_mean: -9.862884963768115
- base_max_mean: 9.104846014492754
- base_abs_mean_mean: 0.5893421276755955
- base_rms_mean: 0.8251033872365952
- base_l2_mean: 517.3970567454462
- base_channel_mean_std_mean: 0.5554052087599817
- base_channel_std_mean_mean: 0.5356448131842889
- base_channel_std_std_mean: 0.26624362073514773
- base_spatial_highfreq_ratio_mean: 0.2776282065804454
- pred_mean_mean: -0.013911786981571308
- pred_std_mean: 0.741718694459701
- pred_min_mean: -9.046761775362318
- pred_max_mean: 8.442538496376812
- pred_abs_mean_mean: 0.5216725576614988
- pred_rms_mean: 0.7421005108002303
- pred_l2_mean: 465.3484965338223
- pred_channel_mean_std_mean: 0.43654480641302856
- pred_channel_std_mean_mean: 0.5301346106697684
- pred_channel_std_std_mean: 0.25548154095430736
- pred_spatial_highfreq_ratio_mean: 0.2400443655965121
- adapter_delta_raw_mean_mean: 0.07185651694991342
- adapter_delta_raw_std_mean: 0.8657158777333688
- adapter_delta_raw_min_mean: -7.064990942028985
- adapter_delta_raw_max_mean: 10.32334125905797
- adapter_delta_raw_abs_mean_mean: 0.5793508965036144
- adapter_delta_raw_rms_mean: 0.868835720948983
- adapter_delta_raw_l2_mean: 544.8202648992124
- adapter_delta_raw_channel_mean_std_mean: 0.7023285037257533
- adapter_delta_raw_channel_std_mean_mean: 0.44819382345978764
- adapter_delta_raw_channel_std_std_mean: 0.21587058407780918
- adapter_delta_raw_spatial_highfreq_ratio_mean: 0.2258102516171293
- base_to_target_l1_mean: 0.537210745278044
- base_to_target_mse_mean: 0.4844576778241258
- base_to_target_rms_mean: 0.6913204865287179
- base_to_target_relative_l2_mean: 0.8761014930795932
- base_to_target_cosine_mean: 0.6220095740982156
- base_to_target_mean_delta_mean: -0.038644577681502
- base_to_target_std_delta_mean: 0.028125818224920742
- pred_to_target_l1_mean: 0.41075400727382605
- pred_to_target_mse_mean: 0.28247161700889684
- pred_to_target_rms_mean: 0.5291065102362115
- pred_to_target_relative_l2_mean: 0.6696491835341938
- pred_to_target_cosine_mean: 0.7572698785343032
- pred_to_target_mean_delta_mean: -0.005392257193392837
- pred_to_target_std_delta_mean: -0.05362168191999629
- pred_to_base_l1_mean: 0.3527805697766767
- pred_to_base_mse_mean: 0.18148479024893133
- pred_to_base_rms_mean: 0.4237551522859629
- pred_to_base_relative_l2_mean: 0.5183364163706268
- pred_to_base_cosine_mean: 0.8526587746497514
- pred_to_base_mean_delta_mean: 0.03325232042951266
- pred_to_base_std_delta_mean: -0.08174750014491704
- pred_l1_improvement_vs_base_mean: 0.12645673800421797
- pred_relative_l2_improvement_vs_base_mean: 0.20645230954539948
- count: 552
- actual_payload_bpp_mean: 0.013999109682829483
- pred_condition_l1_win_rate: 0.9963768115942029
- pred_relative_l2_win_rate: 0.9981884057971014
- condition_residual_scale: 0.85
- condition_residual_tanh: True
- condition_delta_ablation: normal
- semantic_latent_ablation: normal
- detail_context_ablation: normal
- ablation_shuffle_seed: 1234
- payload_policy: analysis only; predicted condition is deterministic from decoded CoSER tensors, fixed adapter weights, and fixed CoD-Lite weights, so no image-specific side information is introduced

## Artifacts

- summary: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailcontrast_condition_stats_full552/summary.json`
- per_image_condition_stats: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailcontrast_condition_stats_full552/per_image_condition_stats.jsonl`
- channel_stats: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailcontrast_condition_stats_full552/channel_stats.json`
- activation_histograms: `results/stage4_cod_lite_condition_stats/20260629_stage4_detailcontrast_condition_stats_full552/activation_histograms.json`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_153000-troe2z3e`
