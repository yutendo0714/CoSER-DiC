# 20260627_stage2_token_prior_tf384_l6_12kstep_32768tokens_stage1_lpips002_ft500_lr1e4_do02_es

Date: 2026-06-27T19:33:48

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage2_token_prior.py --tokens outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_32768_oi_div2k_from_stage1_lpips002_ft500_fullroots/semantic_tokens.pt --max-steps 12000 --eval-every 500 --batch-size 64 --num-workers 2 --lr 1e-4 --weight-decay 0.01 --grad-clip-norm 1.0 --d-model 384 --num-layers 6 --num-heads 6 --dropout 0.2 --early-stop-patience 5 --early-stop-min-delta 0.001 --wandb-mode offline --run-name 20260627_stage2_token_prior_tf384_l6_12kstep_32768tokens_stage1_lpips002_ft500_lr1e4_do02_es
```

## Summary

- tokens: outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_32768_oi_div2k_from_stage1_lpips002_ft500_fullroots/semantic_tokens.pt
- checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf384_l6_12kstep_32768tokens_stage1_lpips002_ft500_lr1e4_do02_es.pt
- num_train_images: 29491
- num_val_images: 3277
- context_length: 64
- vocab_size: 8192
- max_steps: 12000
- completed_steps: 9500
- stopped_early: True
- best_step: 7000
- label_smoothing: 0.0
- grad_clip_norm: 1.0
- early_stop_patience: 5
- early_stop_min_delta: 0.001
- max_val_batches: 0
- best_val_bits_per_token: 10.193437909643627
- final_val_bits_per_token: 10.229548206153728
- final_val_top1_hit: 0.045201403722917304
- final_val_top5_hit: 0.11584051724137931
- final_val_top64_hit: 0.33018957888312483

## Artifacts

- checkpoint: `checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf384_l6_12kstep_32768tokens_stage1_lpips002_ft500_lr1e4_do02_es.pt`
- summary: `checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf384_l6_12kstep_32768tokens_stage1_lpips002_ft500_lr1e4_do02_es_summary.json`
