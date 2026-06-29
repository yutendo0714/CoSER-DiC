# 20260629_stage4_stats_hf_condition_gate_channel_balanced_ft1200_b8

Date: 2026-06-29T11:53:26

## Command

```bash
scripts/train_stage4_cod_lite_condition_gate.py --adapter-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailfilm_stats_hf_ft1200_b12.pt --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage4_stats_hf_condition_gate_channel_balanced_ft1200_b8 --crop-size 512 --batch-size 8 --grad-accum-steps 1 --num-workers 4 --max-steps 1200 --lr 1e-4 --hidden-channels 192 --num-blocks 3 --init-gate 0.85 --channel-gate --condition-l1-weight 0.02 --condition-regression-guard-weight 0.05 --image-l1-weight 0.8 --lpips-weight 0.08 --dists-weight 0.15 --ms-ssim-weight 0.2 --stage3-l1-guard-weight 1.0 --stage3-mse-guard-weight 2.0 --wandb-mode offline
```

## Summary

- loss_mean: 0.19071989811956883
- image_l1_mean: 0.05774284075635175
- lpips_mean: 0.4136891671270132
- dists_loss_mean: 0.28205654345452785
- ms_ssim_loss_mean: 0.2738552400966485
- condition_l1_mean: 0.4031606362760067
- ungated_condition_l1_mean: 0.3970064852386713
- condition_regression_guard_mean: 0.0061547795683145525
- stage3_l1_guard_mean: 0.0036465379841198833
- stage3_mse_guard_mean: 0.0011667352133372332
- gate_mean_loss_mean: 0.0
- stage4_psnr_mean: 21.30108535607656
- stage3_psnr_mean: 21.967145652770995
- stage4_ms_ssim_mean: 0.7261447599033515
- stage3_ms_ssim_mean: 0.7437610627214114
- gate_mean_mean: 0.9171472977598508
- gate_std_mean: 0.1687594736888362
- gate_min_mean: 0.011405147551174047
- gate_max_mean: 0.99849609375
- gate_mean_over_steps: 0.9171472977598508
- gate_mean_std_over_steps: 0.013700965791940689
- payload_policy: deterministic decoder-side condition gate from decoded CoSER tensors and fixed model weights; no transmitted side information and no RGB output blending

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_condition_gate/20260629_stage4_stats_hf_condition_gate_channel_balanced_ft1200_b8.pt`
- summary: `results/stage4_cod_lite_condition_gate/20260629_stage4_stats_hf_condition_gate_channel_balanced_ft1200_b8/summary.json`
- results_dir: `results/stage4_cod_lite_condition_gate/20260629_stage4_stats_hf_condition_gate_channel_balanced_ft1200_b8`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_114036-tbtquxbb`
