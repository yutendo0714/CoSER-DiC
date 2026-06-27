# 20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_stage1_lpips002_ft500_lr1e4_do02_es

Date: 2026-06-27T18:54:48

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage2_token_prior.py --tokens outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_32768_oi_div2k_from_stage1_lpips002_ft500_fullroots/semantic_tokens.pt --max-steps 8000 --eval-every 500 --batch-size 64 --num-workers 2 --lr 1e-4 --weight-decay 0.01 --grad-clip-norm 1.0 --d-model 256 --num-layers 4 --num-heads 4 --dropout 0.2 --early-stop-patience 4 --early-stop-min-delta 0.001 --wandb-mode offline --run-name 20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_stage1_lpips002_ft500_lr1e4_do02_es
```

## Summary

- tokens: outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_32768_oi_div2k_from_stage1_lpips002_ft500_fullroots/semantic_tokens.pt
- checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_stage1_lpips002_ft500_lr1e4_do02_es.pt
- num_train_images: 29491
- num_val_images: 3277
- context_length: 64
- vocab_size: 8192
- max_steps: 8000
- completed_steps: 8000
- stopped_early: False
- best_step: 8000
- label_smoothing: 0.0
- grad_clip_norm: 1.0
- early_stop_patience: 4
- early_stop_min_delta: 0.001
- max_val_batches: 0
- best_val_bits_per_token: 10.343893166816136
- final_val_bits_per_token: 10.343893166816136
- final_val_top1_hit: 0.0391697818126335
- final_val_top5_hit: 0.10366283948733598
- final_val_top64_hit: 0.3054241684467501

## Artifacts

- checkpoint: `checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_stage1_lpips002_ft500_lr1e4_do02_es.pt`
- summary: `checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_stage1_lpips002_ft500_lr1e4_do02_es_summary.json`
