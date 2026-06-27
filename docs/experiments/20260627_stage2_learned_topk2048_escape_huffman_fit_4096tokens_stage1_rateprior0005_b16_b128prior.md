# 20260627_stage2_learned_topk2048_escape_huffman_fit_4096tokens_stage1_rateprior0005_b16_b128prior

Date: 2026-06-27T23:53:48

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/fit_stage2_learned_topk_escape_prior.py --tokens outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_rateprior0005_b16_fullroots/semantic_tokens.pt --token-prior-checkpoint checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_3kstep_4096tokens_stage1_rateprior0005_b16_lr1e4_do02_b128_es.pt --topk 2048 --event-smoothing-count 1 --batch-size 128 --num-workers 2 --image-size 256 --wandb-mode offline --run-name 20260627_stage2_learned_topk2048_escape_huffman_fit_4096tokens_stage1_rateprior0005_b16_b128prior
```

## Summary

- tokens: outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_rateprior0005_b16_fullroots/semantic_tokens.pt
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_3kstep_4096tokens_stage1_rateprior0005_b16_lr1e4_do02_b128_es.pt
- prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk2048_escape_huffman_fit_4096tokens_stage1_rateprior0005_b16_b128prior/learned_topk_escape_huffman_prior.json
- num_images: 4096
- token_shape: [8, 8]
- codebook_size: 8192
- topk: 2048
- event_smoothing_count: 1
- total_tokens: 262144
- top1_hit_rate: 0.0796356201171875
- top5_hit_rate: 0.19281387329101562
- topk_hit_rate: 0.9568634033203125
- escape_rate: 0.0431365966796875
- event_entropy_bits: 8.32220928244565
- mean_unpadded_bits_per_token: 8.915935516357422
- mean_payload_bytes_per_image: 71.7626953125
- mean_payload_bpp_image_size: 0.00876009464263916
- fixed_bits_payload_bpp_image_size: 0.0126953125

## Artifacts

- prior: `outputs/stage2_learned_entropy/20260627_stage2_learned_topk2048_escape_huffman_fit_4096tokens_stage1_rateprior0005_b16_b128prior/learned_topk_escape_huffman_prior.json`
- summary: `outputs/stage2_learned_entropy/20260627_stage2_learned_topk2048_escape_huffman_fit_4096tokens_stage1_rateprior0005_b16_b128prior/summary.json`
