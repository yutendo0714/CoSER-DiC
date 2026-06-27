# 20260627_stage2_token_prior_tf256_l4_5kstep_4096tokens_unscaleclip_probe

Date: 2026-06-27T10:17:01

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/train_stage2_token_prior.py --tokens outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_best/semantic_tokens.pt --max-steps 5000 --eval-every 500 --batch-size 64 --num-workers 2 --d-model 256 --num-layers 4 --num-heads 4 --lr 3e-4 --weight-decay 0.01 --amp --wandb-mode offline --run-name 20260627_stage2_token_prior_tf256_l4_5kstep_4096tokens_unscaleclip_probe
```

## Summary

- tokens: outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_best/semantic_tokens.pt
- checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_5kstep_4096tokens_unscaleclip_probe.pt
- num_train_images: 3686
- num_val_images: 410
- context_length: 64
- vocab_size: 8192
- max_steps: 5000
- best_val_bits_per_token: 11.272778148943589
- final_val_bits_per_token: 18.919609579222765
- final_val_top1_hit: 0.015091463414634146
- final_val_top5_hit: 0.04470274390243902
- final_val_top64_hit: 0.1576981707317073

## Artifacts

- checkpoint: `checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_5kstep_4096tokens_unscaleclip_probe.pt`
- summary: `checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_5kstep_4096tokens_unscaleclip_probe_summary.json`
