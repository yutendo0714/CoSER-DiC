# 20260628_stage2_learned_topk2048_escape_huffman_fit_32768tokens_stage1_rateprior0005_b16_tf384_l6_12k

Date: 2026-06-28T01:20:05

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/fit_stage2_learned_topk_escape_prior.py --tokens outputs/stage2_semantic_tokens/20260628_stage2_semantic_tokens_32768_oi_div2k_from_stage1_rateprior0005_b16_fullroots/semantic_tokens.pt --token-prior-checkpoint checkpoints/stage2_token_prior/20260628_stage2_token_prior_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_es.pt --topk 2048 --max-images 32768 --batch-size 64 --num-workers 2 --image-size 256 --wandb-mode offline --run-name 20260628_stage2_learned_topk2048_escape_huffman_fit_32768tokens_stage1_rateprior0005_b16_tf384_l6_12k
```

## Summary

- tokens: outputs/stage2_semantic_tokens/20260628_stage2_semantic_tokens_32768_oi_div2k_from_stage1_rateprior0005_b16_fullroots/semantic_tokens.pt
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260628_stage2_token_prior_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_es.pt
- prior: outputs/stage2_learned_entropy/20260628_stage2_learned_topk2048_escape_huffman_fit_32768tokens_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json
- num_images: 32768
- token_shape: [8, 8]
- codebook_size: 8192
- topk: 2048
- event_smoothing_count: 1
- total_tokens: 2097152
- top1_hit_rate: 0.05965757369995117
- top5_hit_rate: 0.15061044692993164
- topk_hit_rate: 0.9234104156494141
- escape_rate: 0.07658958435058594
- event_entropy_bits: 8.76026394929277
- mean_unpadded_bits_per_token: 9.782552719116211
- mean_payload_bytes_per_image: 78.69845581054688
- mean_payload_bpp_image_size: 0.009606745094060898
- fixed_bits_payload_bpp_image_size: 0.0126953125

## Artifacts

- prior: `outputs/stage2_learned_entropy/20260628_stage2_learned_topk2048_escape_huffman_fit_32768tokens_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json`
- summary: `outputs/stage2_learned_entropy/20260628_stage2_learned_topk2048_escape_huffman_fit_32768tokens_stage1_rateprior0005_b16_tf384_l6_12k/summary.json`
