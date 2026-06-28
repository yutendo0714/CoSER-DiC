# 20260628_stage2_token_prior_crop512_tf384_l6_b64_memory_probe

Date: 2026-06-28T01:58:14

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage2_token_prior.py --tokens outputs/stage2_semantic_tokens/20260628_stage2_semantic_tokens_32768_oi_val_crop512_stage1_rateprior0005_b16/semantic_tokens.pt --max-steps 1 --eval-every 1 --batch-size 64 --num-workers 2 --lr 1e-4 --d-model 384 --num-layers 6 --num-heads 6 --dropout 0.2 --amp --wandb-mode offline --run-name 20260628_stage2_token_prior_crop512_tf384_l6_b64_memory_probe
```

## Summary

- tokens: outputs/stage2_semantic_tokens/20260628_stage2_semantic_tokens_32768_oi_val_crop512_stage1_rateprior0005_b16/semantic_tokens.pt
- checkpoint: checkpoints/stage2_token_prior/20260628_stage2_token_prior_crop512_tf384_l6_b64_memory_probe.pt
- num_train_images: 29491
- num_val_images: 3277
- context_length: 256
- vocab_size: 8192
- max_steps: 1
- completed_steps: 1
- stopped_early: False
- best_step: 1
- label_smoothing: 0.0
- grad_clip_norm: 1.0
- early_stop_patience: 0
- early_stop_min_delta: 0.0
- max_val_batches: 0
- best_val_bits_per_token: 13.21961097369931
- final_val_bits_per_token: 13.21961097369931
- final_val_top1_hit: 0.00013827433628318584
- final_val_top5_hit: 0.0007009078425389075
- final_val_top64_hit: 0.008466919057064388

## Artifacts

- checkpoint: `checkpoints/stage2_token_prior/20260628_stage2_token_prior_crop512_tf384_l6_b64_memory_probe.pt`
- summary: `checkpoints/stage2_token_prior/20260628_stage2_token_prior_crop512_tf384_l6_b64_memory_probe_summary.json`
