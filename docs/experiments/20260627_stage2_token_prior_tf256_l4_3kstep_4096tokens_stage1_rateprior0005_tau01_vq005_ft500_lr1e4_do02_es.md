# 20260627_stage2_token_prior_tf256_l4_3kstep_4096tokens_stage1_rateprior0005_tau01_vq005_ft500_lr1e4_do02_es

Date: 2026-06-27T22:33:28

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage2_token_prior.py --tokens outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_rateprior0005_tau01_vq005_ft500_fullroots/semantic_tokens.pt --batch-size 64 --num-workers 2 --max-steps 3000 --eval-every 500 --lr 1e-4 --weight-decay 0.01 --d-model 256 --num-layers 4 --num-heads 4 --dropout 0.2 --grad-clip-norm 1.0 --early-stop-patience 2 --early-stop-min-delta 0.001 --wandb-mode offline --run-name 20260627_stage2_token_prior_tf256_l4_3kstep_4096tokens_stage1_rateprior0005_tau01_vq005_ft500_lr1e4_do02_es
```

## Summary

- tokens: outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_rateprior0005_tau01_vq005_ft500_fullroots/semantic_tokens.pt
- checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_3kstep_4096tokens_stage1_rateprior0005_tau01_vq005_ft500_lr1e4_do02_es.pt
- num_train_images: 3686
- num_val_images: 410
- context_length: 64
- vocab_size: 8192
- max_steps: 3000
- completed_steps: 3000
- stopped_early: False
- best_step: 2500
- label_smoothing: 0.0
- grad_clip_norm: 1.0
- early_stop_patience: 2
- early_stop_min_delta: 0.001
- max_val_batches: 0
- best_val_bits_per_token: 11.624004507852247
- final_val_bits_per_token: 11.66606306814417
- final_val_top1_hit: 0.0265625
- final_val_top5_hit: 0.06413871951219512
- final_val_top64_hit: 0.18921493902439024

## Artifacts

- checkpoint: `checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_3kstep_4096tokens_stage1_rateprior0005_tau01_vq005_ft500_lr1e4_do02_es.pt`
- summary: `checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_3kstep_4096tokens_stage1_rateprior0005_tau01_vq005_ft500_lr1e4_do02_es_summary.json`
