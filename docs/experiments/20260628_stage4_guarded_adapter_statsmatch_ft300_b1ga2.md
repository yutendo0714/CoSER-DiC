# 20260628_stage4_guarded_adapter_statsmatch_ft300_b1ga2

Date: 2026-06-28T23:35:50

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_traincache_clic_oi_2048_seed20260628/reconstructions/manifest.jsonl --run-name 20260628_stage4_guarded_adapter_statsmatch_ft300_b1ga2 --crop-size 512 --batch-size 1 --grad-accum-steps 2 --num-workers 2 --max-steps 300 --lr 2e-5 --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_statsmatch_ft2k_b8.pt --condition-residual-scale 0.75 --condition-residual-tanh --condition-l1-weight 1.0 --condition-cosine-weight 0.25 --condition-channel-stats-weight 0.20 --condition-highfreq-weight 0.05 --image-l1-weight 0.10 --lpips-weight 0.01 --ms-ssim-weight 0.10 --stage3-l1-guard-weight 0.75 --stage3-mse-guard-weight 1.50 --grad-clip-norm 1.0 --save-sample-every 100 --wandb-mode offline
```

## Summary

- loss_mean: 0.5282659873863061
- condition_l1_mean: 0.38442147716879843
- condition_cosine_loss_mean: 0.21039087901512782
- condition_channel_stats_loss_mean: 0.22715943617125353
- condition_highfreq_loss_mean: 0.05320646189774076
- base_condition_l1_mean: 0.519652333209912
- condition_l1_delta_vs_base_mean: -0.13523085604111354
- condition_cosine_mean: 0.7896091209848722
- pred_condition_std_mean: 0.7484706110755602
- target_condition_std_mean: 0.8091431796550751
- pred_condition_highfreq_mean: 0.2233572247127692
- target_condition_highfreq_mean: 0.2436196932196617
- image_l1_mean: 0.05848647509391109
- lpips_mean: 0.40416528257230916
- ms_ssim_loss_mean: 0.27869255463282266
- stage3_l1_guard_mean: 0.004330322071909904
- stage3_mse_guard_mean: 0.001431522869146041
- stage4_psnr_mean: 21.116098486582437
- stage3_psnr_mean: 21.912851428985597
- condition_residual_l1_mean: 0.34421938923497997
- condition_delta_raw_l1_mean: 0.6622234667340915

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260628_stage4_guarded_adapter_statsmatch_ft300_b1ga2.pt`
- summary: `results/stage4_cod_lite_adapter/20260628_stage4_guarded_adapter_statsmatch_ft300_b1ga2/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260628_stage4_guarded_adapter_statsmatch_ft300_b1ga2`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260628_233453-rztbqrff`
