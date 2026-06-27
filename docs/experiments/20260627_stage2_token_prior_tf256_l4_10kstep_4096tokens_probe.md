# 20260627_stage2_token_prior_tf256_l4_10kstep_4096tokens_probe

Date: 2026-06-27T10:15:09

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage2_token_prior.py --tokens outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_best/semantic_tokens.pt --max-steps 10000 --eval-every 1000 --batch-size 64 --num-workers 2 --d-model 256 --num-layers 4 --num-heads 4 --lr 3e-4 --weight-decay 0.01 --amp --wandb-mode offline --run-name 20260627_stage2_token_prior_tf256_l4_10kstep_4096tokens_probe
```

## Summary

- tokens: outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_best/semantic_tokens.pt
- checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_10kstep_4096tokens_probe.pt
- num_train_images: 3686
- num_val_images: 410
- context_length: 64
- vocab_size: 8192
- max_steps: 10000
- best_val_bits_per_token: 11.27958991026012
- final_val_bits_per_token: 12.207110380212333
- final_val_top1_hit: 0.025152439024390245
- final_val_top5_hit: 0.06833079268292683
- final_val_top64_hit: 0.20423018292682926

## Artifacts

- checkpoint: `checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_10kstep_4096tokens_probe.pt`
- summary: `checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_10kstep_4096tokens_probe_summary.json`
