# 20260627_stage2_learned_topk64_escape_huffman_fit_32768tokens_8kprior

Date: 2026-06-27T10:51:53

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/fit_stage2_learned_topk_escape_prior.py --tokens outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_32768_oi_div2k_from_stage1_best/semantic_tokens.pt --token-prior-checkpoint checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt --topk 64 --event-smoothing-count 1 --batch-size 128 --num-workers 4 --wandb-mode offline --run-name 20260627_stage2_learned_topk64_escape_huffman_fit_32768tokens_8kprior
```

## Summary

- tokens: outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_32768_oi_div2k_from_stage1_best/semantic_tokens.pt
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt
- prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk64_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json
- num_images: 32768
- token_shape: [8, 8]
- codebook_size: 8192
- topk: 64
- event_smoothing_count: 1
- total_tokens: 2097152
- top1_hit_rate: 0.05962562561035156
- top5_hit_rate: 0.15297794342041016
- topk_hit_rate: 0.41835737228393555
- escape_rate: 0.5816426277160645
- event_entropy_bits: 3.2088943044014804
- mean_unpadded_bits_per_token: 10.801526546478271
- mean_payload_bytes_per_image: 86.84805297851562
- mean_payload_bpp_image_size: 0.010601568967103958
- fixed_bits_payload_bpp_image_size: 0.0126953125

## Artifacts

- prior: `outputs/stage2_learned_entropy/20260627_stage2_learned_topk64_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json`
- summary: `outputs/stage2_learned_entropy/20260627_stage2_learned_topk64_escape_huffman_fit_32768tokens_8kprior/summary.json`
