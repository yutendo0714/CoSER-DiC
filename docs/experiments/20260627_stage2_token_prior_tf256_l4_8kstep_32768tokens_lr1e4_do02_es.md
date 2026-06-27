# 20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es

Date: 2026-06-27T10:44:48

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage2_token_prior.py --tokens outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_32768_oi_div2k_from_stage1_best/semantic_tokens.pt --max-steps 8000 --eval-every 500 --batch-size 128 --num-workers 4 --d-model 256 --num-layers 4 --num-heads 4 --lr 1e-4 --weight-decay 0.05 --dropout 0.2 --grad-clip-norm 1.0 --early-stop-patience 4 --early-stop-min-delta 0.001 --amp --wandb-mode offline --run-name 20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es
```

## Summary

- tokens: outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_32768_oi_div2k_from_stage1_best/semantic_tokens.pt
- checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt
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
- best_val_bits_per_token: 9.883719817893285
- final_val_bits_per_token: 9.883719817893285
- final_val_top1_hit: 0.049392546536466284
- final_val_top5_hit: 0.1294009383582545
- final_val_top64_hit: 0.36146341928593223

## Artifacts

- checkpoint: `checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt`
- summary: `checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es_summary.json`
