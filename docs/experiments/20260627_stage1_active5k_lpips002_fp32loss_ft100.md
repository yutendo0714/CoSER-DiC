# 20260627_stage1_active5k_lpips002_fp32loss_ft100

Date: 2026-06-27T17:56:32

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage1_semantic_vq.py --init-checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --fixed-quantize-mix 1 --loss-l1-sem 1.0 --loss-ms-ssim-sem 0 --loss-lpips-sem 0.02 --loss-vq 0.05 --loss-codebook-usage 0.005 --grad-clip-norm 1.0 --lr 1e-5 --max-steps 100 --batch-size 4 --num-workers 4 --log-every 20 --sample-every 50 --wandb-mode offline --run-name 20260627_stage1_active5k_lpips002_fp32loss_ft100
```

## Summary

- loss_total: 0.12616018950939178
- loss_l1_sem: 0.05938854068517685
- loss_ms_ssim_sem: 0.0
- loss_vq: 1.0440404415130615
- loss_vq_commitment: 0.8352323770523071
- loss_vq_codebook: 0.8352323770523071
- perplexity: 213.08070373535156
- soft_perplexity: 229.232177734375
- assignment_sample_entropy_bits: 0.15597452223300934
- assignment_avg_entropy_bits: 7.735256195068359
- soft_usage_entropy_bits: 7.840665817260742
- dead_code_ratio: 0.97265625
- used_codes: 225.0
- usage_loss: 0.3968719244003296
- quantize_mix: 1.0
- vq_scale: 1.0
- usage_scale: 1.0
- psnr_sem: 20.427451133728027
- x_sem_mean: 0.444580078125
- x_sem_std: 0.2169189453125
- x_sem_min: 0.0299835205078125
- x_sem_max: 0.9716796875
- x_sem_r_mean: 0.4697265625
- x_sem_g_mean: 0.4462890625
- x_sem_b_mean: 0.41748046875
- h_mean: 0.0254058837890625
- h_std: 2.20703125
- h_absmax: 24.5
- continuous_std: 2.20703125
- quantized_std: 2.037109375
- lr: 1e-05
- step: 100
- grad_norm: 0.384372353553772

## Artifacts

- checkpoint: `checkpoints/stage1_semantic_vq/20260627_stage1_active5k_lpips002_fp32loss_ft100.pt`
- output_dir: `outputs/stage1_semantic_vq/20260627_stage1_active5k_lpips002_fp32loss_ft100`
- summary: `outputs/stage1_semantic_vq/20260627_stage1_active5k_lpips002_fp32loss_ft100/summary.json`
