# 20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior

Date: 2026-06-27T11:14:51

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/fit_stage2_learned_topk_escape_prior.py --tokens outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_32768_oi_div2k_from_stage1_best/semantic_tokens.pt --token-prior-checkpoint checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt --topk 512 --event-smoothing-count 1 --batch-size 128 --num-workers 4 --wandb-mode offline --run-name 20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior
```

## Summary

- tokens: outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_32768_oi_div2k_from_stage1_best/semantic_tokens.pt
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt
- prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json
- num_images: 32768
- token_shape: [8, 8]
- codebook_size: 8192
- topk: 512
- event_smoothing_count: 1
- total_tokens: 2097152
- top1_hit_rate: 0.05962562561035156
- top5_hit_rate: 0.15297794342041016
- topk_hit_rate: 0.7337985038757324
- escape_rate: 0.2662014961242676
- event_entropy_bits: 6.500670435906907
- mean_unpadded_bits_per_token: 9.981475830078125
- mean_payload_bytes_per_image: 80.29168701171875
- mean_payload_bpp_image_size: 0.009801231324672699
- fixed_bits_payload_bpp_image_size: 0.0126953125

## Artifacts

- prior: `outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json`
- summary: `outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/summary.json`
