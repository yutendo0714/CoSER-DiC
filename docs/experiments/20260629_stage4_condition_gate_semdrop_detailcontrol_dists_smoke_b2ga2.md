# 20260629_stage4_condition_gate_semdrop_detailcontrol_dists_smoke_b2ga2

Date: 2026-06-29T02:59:39

## Command

```bash
scripts/train_stage4_cod_lite_condition_gate.py --adapter-checkpoint checkpoints/stage4_cod_lite_adapter/20260629_stage4_fast8192_detailcontrol_semdrop020_ft800_b8.pt --manifest results/bitstreams/stage3_training_cache_fast/20260629_stage3_traincache_fast_clic_oi_8192_seed20260629_b4/reconstructions/manifest.jsonl --run-name 20260629_stage4_condition_gate_semdrop_detailcontrol_dists_smoke_b2ga2 --crop-size 512 --batch-size 2 --grad-accum-steps 2 --num-workers 2 --max-steps 20 --lr 1e-4 --hidden-channels 128 --num-blocks 2 --gate-min 0.55 --gate-max 1.0 --init-gate 0.85 --condition-l1-weight 0.1 --condition-regression-guard-weight 0.1 --image-l1-weight 0.25 --lpips-weight 0.08 --dists-weight 0.4 --ms-ssim-weight 0.1 --stage3-l1-guard-weight 0.3 --stage3-mse-guard-weight 0.8 --gate-mean-target 0.85 --gate-mean-weight 0.05 --grad-clip-norm 1.0 --wandb-mode offline
```

## Summary

- loss_mean: 0.22854833863675594
- image_l1_mean: 0.057342273136600855
- lpips_mean: 0.410176919400692
- dists_loss_mean: 0.27911070436239244
- ms_ssim_loss_mean: 0.271199943125248
- condition_l1_mean: 0.39659354090690613
- ungated_condition_l1_mean: 0.3933980233967304
- condition_regression_guard_mean: 0.0038946397602558137
- stage3_l1_guard_mean: 0.004493828839622438
- stage3_mse_guard_mean: 0.001529533852590248
- gate_mean_loss_mean: 0.000274921696632191
- stage4_psnr_mean: 21.470658159255983
- stage3_psnr_mean: 22.320216178894043
- stage4_ms_ssim_mean: 0.728800056874752
- stage3_ms_ssim_mean: 0.7506360486149788
- gate_mean_mean: 0.8612739562988281
- gate_std_mean: 0.003888856657431461
- gate_min_mean: 0.85302734375
- gate_max_mean: 0.87177734375
- gate_mean_over_steps: 0.8612739562988281
- gate_mean_std_over_steps: 0.01199908647686243
- payload_policy: deterministic decoder-side condition gate from decoded CoSER tensors and fixed model weights; no transmitted side information and no RGB output blending

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_condition_gate/20260629_stage4_condition_gate_semdrop_detailcontrol_dists_smoke_b2ga2.pt`
- summary: `results/stage4_cod_lite_condition_gate/20260629_stage4_condition_gate_semdrop_detailcontrol_dists_smoke_b2ga2/summary.json`
- results_dir: `results/stage4_cod_lite_condition_gate/20260629_stage4_condition_gate_semdrop_detailcontrol_dists_smoke_b2ga2`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_025931-5kia8osm`
