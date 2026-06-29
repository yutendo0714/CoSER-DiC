# 20260628_stage4_condition_stats_full552_ablate_detailzero

Date: 2026-06-28T23:30:53

## Command

```bash
scripts/analyze_stage4_cod_lite_condition_stats.py --checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_statsmatch_ft2k_b8.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/reconstructions/manifest.jsonl --per-image-metrics results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_full552_feature_cache/per_image_metrics.jsonl --run-name 20260628_stage4_condition_stats_full552_ablate_detailzero --crop-size 512 --batch-size 1 --num-workers 2 --detail-context-ablation zero --wandb-mode offline
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
- pred_mean_mean: -0.014123061964934437
- pred_std_mean: 0.7270007400193076
- pred_min_mean: -9.152909873188406
- pred_max_mean: 8.487686820652174
- pred_abs_mean_mean: 0.5050123925118343
- pred_rms_mean: 0.7273861097468846
- pred_l2_mean: 456.12154504527217
- pred_channel_mean_std_mean: 0.43815845623612404
- pred_channel_std_mean_mean: 0.5048551243068515
- pred_channel_std_std_mean: 0.26165847713802604
- pred_spatial_highfreq_ratio_mean: 0.2360248340536719
- adapter_delta_raw_mean_mean: 0.0791502842321938
- adapter_delta_raw_std_mean: 0.9191059447501017
- adapter_delta_raw_min_mean: -7.027060688405797
- adapter_delta_raw_max_mean: 8.864526721014492
- adapter_delta_raw_abs_mean_mean: 0.6419731904415117
- adapter_delta_raw_rms_mean: 0.9226185055314631
- adapter_delta_raw_l2_mean: 578.545798426089
- adapter_delta_raw_channel_mean_std_mean: 0.7437695175193358
- adapter_delta_raw_channel_std_mean_mean: 0.4931518065108769
- adapter_delta_raw_channel_std_std_mean: 0.20198236355908972
- adapter_delta_raw_spatial_highfreq_ratio_mean: 0.23419949325962344
- base_to_target_l1_mean: 0.5370939215780168
- base_to_target_mse_mean: 0.4842399032381566
- base_to_target_rms_mean: 0.6911686438581218
- base_to_target_relative_l2_mean: 0.8758228153422258
- base_to_target_cosine_mean: 0.6221415235836437
- base_to_target_mean_delta_mean: -0.03866021500832268
- base_to_target_std_delta_mean: 0.027971822282542354
- pred_to_target_l1_mean: 0.41751222269258637
- pred_to_target_mse_mean: 0.2927110103377398
- pred_to_target_rms_mean: 0.5385050471818101
- pred_to_target_relative_l2_mean: 0.681455966193175
- pred_to_target_cosine_mean: 0.744556214498437
- pred_to_target_mean_delta_mean: -0.005617385144050522
- pred_to_target_std_delta_mean: -0.0684170227335847
- pred_to_base_l1_mean: 0.33852839421318925
- pred_to_base_mse_mean: 0.16176015253354242
- pred_to_base_rms_mean: 0.40095005832288577
- pred_to_base_relative_l2_mean: 0.49021223792131396
- pred_to_base_cosine_mean: 0.8690212409997332
- pred_to_base_mean_delta_mean: 0.03304282999158823
- pred_to_base_std_delta_mean: -0.09638884501612704
- pred_l1_improvement_vs_base_mean: 0.11958169888543047
- pred_relative_l2_improvement_vs_base_mean: 0.19436684914905092
- count: 552
- actual_payload_bpp_mean: 0.013999109682829483
- pred_condition_l1_win_rate: 0.9981884057971014
- pred_relative_l2_win_rate: 0.9981884057971014
- condition_residual_scale: 0.75
- condition_residual_tanh: True
- condition_delta_ablation: normal
- semantic_latent_ablation: normal
- detail_context_ablation: zero
- ablation_shuffle_seed: 1234
- payload_policy: analysis only; predicted condition is deterministic from decoded CoSER tensors, fixed adapter weights, and fixed CoD-Lite weights, so no image-specific side information is introduced

## Artifacts

- summary: `results/stage4_cod_lite_condition_stats/20260628_stage4_condition_stats_full552_ablate_detailzero/summary.json`
- per_image_condition_stats: `results/stage4_cod_lite_condition_stats/20260628_stage4_condition_stats_full552_ablate_detailzero/per_image_condition_stats.jsonl`
- channel_stats: `results/stage4_cod_lite_condition_stats/20260628_stage4_condition_stats_full552_ablate_detailzero/channel_stats.json`
- activation_histograms: `results/stage4_cod_lite_condition_stats/20260628_stage4_condition_stats_full552_ablate_detailzero/activation_histograms.json`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_233026-uwvpvdzl`
