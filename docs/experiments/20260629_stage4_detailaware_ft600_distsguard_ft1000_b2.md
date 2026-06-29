# 20260629_stage4_detailaware_ft600_distsguard_ft1000_b2

Date: 2026-06-29T00:41:01

## Command

```bash
scripts/train_stage4_cod_lite_adapter.py --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_traincache_clic_oi_2048_seed20260628/reconstructions/manifest.jsonl --run-name 20260629_stage4_detailaware_ft600_distsguard_ft1000_b2 --crop-size 512 --batch-size 2 --grad-accum-steps 1 --num-workers 2 --max-steps 1000 --lr 1e-5 --adapter-kind pyramid --semantic-channels 256 --detail-context residual_grid_codes --hidden-channels 256 --num-image-blocks 4 --num-condition-blocks 4 --num-detail-blocks 3 --num-fusion-blocks 4 --init-checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_detailaware_adapter_idstart_ft600_b2.pt --condition-residual-scale 0.85 --condition-residual-tanh --condition-l1-weight 0.80 --condition-cosine-weight 0.25 --condition-channel-stats-weight 0.20 --condition-highfreq-weight 0.05 --image-l1-weight 0.10 --lpips-weight 0.04 --dists-weight 0.12 --ms-ssim-weight 0.10 --stage3-l1-guard-weight 1.00 --stage3-mse-guard-weight 2.00 --grad-clip-norm 1.0 --save-sample-every 500 --wandb-mode offline
```

## Summary

- loss_mean: 0.4859567560553551
- condition_l1_mean: 0.3789761669933796
- condition_cosine_loss_mean: 0.20245050275325774
- condition_channel_stats_loss_mean: 0.21364015763998032
- condition_highfreq_loss_mean: 0.05316722065582871
- base_condition_l1_mean: 0.5178670597076416
- condition_l1_delta_vs_base_mean: -0.138890892714262
- condition_cosine_mean: 0.7975494972467423
- pred_condition_std_mean: 0.7542989476919174
- target_condition_std_mean: 0.8104306865334511
- pred_condition_highfreq_mean: 0.2207541445493698
- target_condition_highfreq_mean: 0.24317293351888655
- image_l1_mean: 0.05714063861779869
- lpips_mean: 0.38866010212898255
- dists_loss_mean: 0.26809669759869575
- ms_ssim_loss_mean: 0.2737073330283165
- stage3_l1_guard_mean: 0.003581049725878984
- stage3_mse_guard_mean: 0.001196473220945336
- stage4_psnr_mean: 21.380585144996644
- stage3_psnr_mean: 22.060063849449158
- condition_residual_l1_mean: 0.3583987517803907
- condition_delta_raw_l1_mean: 0.5766096567213536

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailaware_ft600_distsguard_ft1000_b2.pt`
- summary: `results/stage4_cod_lite_adapter/20260629_stage4_detailaware_ft600_distsguard_ft1000_b2/summary.json`
- results_dir: `results/stage4_cod_lite_adapter/20260629_stage4_detailaware_ft600_distsguard_ft1000_b2`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_003805-uvzz5smb`
