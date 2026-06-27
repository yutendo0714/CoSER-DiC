# 20260627_stage2_learned_topk64_escape_huffman_fit_4096tokens_3kprior

Date: 2026-06-27T10:27:15

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/fit_stage2_learned_topk_escape_prior.py --tokens outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_best/semantic_tokens.pt --token-prior-checkpoint checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_3kstep_4096tokens_probe.pt --topk 64 --event-smoothing-count 1 --batch-size 128 --num-workers 2 --wandb-mode offline --run-name 20260627_stage2_learned_topk64_escape_huffman_fit_4096tokens_3kprior
```

## Summary

- tokens: outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_best/semantic_tokens.pt
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_3kstep_4096tokens_probe.pt
- prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk64_escape_huffman_fit_4096tokens_3kprior/learned_topk_escape_huffman_prior.json
- num_images: 4096
- token_shape: [8, 8]
- codebook_size: 8192
- topk: 64
- event_smoothing_count: 1
- total_tokens: 262144
- top1_hit_rate: 0.061981201171875
- top5_hit_rate: 0.1530914306640625
- topk_hit_rate: 0.4169502258300781
- escape_rate: 0.5830497741699219
- event_entropy_bits: 3.2027131267711715
- mean_unpadded_bits_per_token: 10.815509796142578
- mean_payload_bytes_per_image: 86.959716796875
- mean_payload_bpp_image_size: 0.01061519980430603
- fixed_bits_payload_bpp_image_size: 0.0126953125

## Artifacts

- prior: `outputs/stage2_learned_entropy/20260627_stage2_learned_topk64_escape_huffman_fit_4096tokens_3kprior/learned_topk_escape_huffman_prior.json`
- summary: `outputs/stage2_learned_entropy/20260627_stage2_learned_topk64_escape_huffman_fit_4096tokens_3kprior/summary.json`
