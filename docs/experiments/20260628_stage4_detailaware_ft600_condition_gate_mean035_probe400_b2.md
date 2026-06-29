# 20260628_stage4_detailaware_ft600_condition_gate_mean035_probe400_b2

Date: 2026-06-29T00:33:26

## Command

```bash
scripts/train_stage4_cod_lite_condition_gate.py --adapter-checkpoint checkpoints/stage4_cod_lite_adapter/20260628_stage4_detailaware_adapter_idstart_ft600_b2.pt --manifest results/bitstreams/stage3_uniform_residual/20260628_stage3_gencodec512_traincache_clic_oi_2048_seed20260628/reconstructions/manifest.jsonl --run-name 20260628_stage4_detailaware_ft600_condition_gate_mean035_probe400_b2 --crop-size 512 --batch-size 2 --grad-accum-steps 1 --num-workers 2 --max-steps 400 --lr 5e-5 --hidden-channels 160 --num-blocks 2 --gate-min 0.0 --gate-max 0.70 --init-gate 0.35 --condition-l1-weight 0.10 --condition-regression-guard-weight 0.50 --image-l1-weight 0.85 --lpips-weight 0.04 --ms-ssim-weight 0.30 --stage3-l1-guard-weight 2.40 --stage3-mse-guard-weight 4.80 --gate-mean-target 0.35 --gate-mean-weight 0.25 --wandb-mode offline
```

## Summary

- loss_mean: 0.23339875916019082
- image_l1_mean: 0.05754606747068465
- lpips_mean: 0.4663444820046425
- ms_ssim_loss_mean: 0.27524767234921454
- condition_l1_mean: 0.41420993976294995
- ungated_condition_l1_mean: 0.37906696148216723
- condition_regression_guard_mean: 0.03514658413827419
- stage3_l1_guard_mean: 0.0040261273889336735
- stage3_mse_guard_mean: 0.0011237241963681299
- gate_mean_loss_mean: 0.03682258968706078
- stage4_psnr_mean: 21.376724514961243
- stage3_psnr_mean: 22.04890119075775
- stage4_ms_ssim_mean: 0.7247523275017739
- stage3_ms_ssim_mean: 0.7428898833692074
- gate_mean_mean: 0.5352267122268677
- gate_std_mean: 0.04628553154383553
- gate_min_mean: 0.379541015625
- gate_max_mean: 0.6103564453125
- gate_mean_over_steps: 0.5352267122268677
- gate_mean_std_over_steps: 0.05013633891940117
- payload_policy: deterministic decoder-side condition gate from decoded CoSER tensors and fixed model weights; no transmitted side information and no RGB output blending

## Artifacts

- checkpoint: `checkpoints/stage4_cod_lite_condition_gate/20260628_stage4_detailaware_ft600_condition_gate_mean035_probe400_b2.pt`
- summary: `results/stage4_cod_lite_condition_gate/20260628_stage4_detailaware_ft600_condition_gate_mean035_probe400_b2/summary.json`
- results_dir: `results/stage4_cod_lite_condition_gate/20260628_stage4_detailaware_ft600_condition_gate_mean035_probe400_b2`
- wandb: `/workspace/CoSER-DiC/wandb/offline-run-20260629_003232-ygli6ivc`
