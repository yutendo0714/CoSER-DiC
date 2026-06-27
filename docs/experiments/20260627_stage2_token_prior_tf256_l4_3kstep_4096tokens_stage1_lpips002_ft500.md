# 20260627_stage2_token_prior_tf256_l4_3kstep_4096tokens_stage1_lpips002_ft500

Date: 2026-06-27T18:06:24

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage2_token_prior.py --tokens outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_lpips002_ft500_fullroots/semantic_tokens.pt --max-steps 3000 --eval-every 500 --batch-size 64 --num-workers 2 --lr 3e-4 --weight-decay 0.01 --grad-clip-norm 1.0 --d-model 256 --num-layers 4 --num-heads 4 --dropout 0.1 --early-stop-patience 0 --wandb-mode offline --run-name 20260627_stage2_token_prior_tf256_l4_3kstep_4096tokens_stage1_lpips002_ft500
```

## Summary

- tokens: outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_lpips002_ft500_fullroots/semantic_tokens.pt
- checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_3kstep_4096tokens_stage1_lpips002_ft500.pt
- num_train_images: 3686
- num_val_images: 410
- context_length: 64
- vocab_size: 8192
- max_steps: 3000
- completed_steps: 3000
- stopped_early: False
- best_step: 1000
- label_smoothing: 0.0
- grad_clip_norm: 1.0
- early_stop_patience: 0
- early_stop_min_delta: 0.0
- max_val_batches: 0
- best_val_bits_per_token: 11.790870512097573
- final_val_bits_per_token: 15.745498580280941
- final_val_top1_hit: 0.014176829268292683
- final_val_top5_hit: 0.037728658536585365
- final_val_top64_hit: 0.1376905487804878

## Artifacts

- checkpoint: `checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_3kstep_4096tokens_stage1_lpips002_ft500.pt`
- summary: `checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_3kstep_4096tokens_stage1_lpips002_ft500_summary.json`
