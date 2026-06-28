# 20260628_stage2_token_prior_crop512_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_amp_es

Date: 2026-06-28T02:09:48

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage2_token_prior.py --tokens outputs/stage2_semantic_tokens/20260628_stage2_semantic_tokens_32768_oi_val_crop512_stage1_rateprior0005_b16/semantic_tokens.pt --max-steps 12000 --eval-every 500 --batch-size 64 --num-workers 4 --lr 1e-4 --weight-decay 0.01 --d-model 384 --num-layers 6 --num-heads 6 --dropout 0.2 --amp --early-stop-patience 5 --early-stop-min-delta 0.002 --wandb-mode offline --run-name 20260628_stage2_token_prior_crop512_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_amp_es
```

## Summary

- tokens: outputs/stage2_semantic_tokens/20260628_stage2_semantic_tokens_32768_oi_val_crop512_stage1_rateprior0005_b16/semantic_tokens.pt
- checkpoint: checkpoints/stage2_token_prior/20260628_stage2_token_prior_crop512_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_amp_es.pt
- num_train_images: 29491
- num_val_images: 3277
- context_length: 256
- vocab_size: 8192
- max_steps: 12000
- completed_steps: 12000
- stopped_early: False
- best_step: 11500
- label_smoothing: 0.0
- grad_clip_norm: 1.0
- early_stop_patience: 5
- early_stop_min_delta: 0.002
- max_val_batches: 0
- best_val_bits_per_token: 8.79940414990234
- final_val_bits_per_token: 8.799048278383921
- final_val_top1_hit: 0.08848365501983521
- final_val_top5_hit: 0.2019460920811718
- final_val_top64_hit: 0.47670911847726577

## Artifacts

- checkpoint: `checkpoints/stage2_token_prior/20260628_stage2_token_prior_crop512_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_amp_es.pt`
- summary: `checkpoints/stage2_token_prior/20260628_stage2_token_prior_crop512_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_amp_es_summary.json`
