# 20260628_stage2_learned_topk512_escape_huffman_fit_32768tokens_stage1_rateprior0005_b16_tf384_l6_12k

Date: 2026-06-28T00:21:04

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/fit_stage2_learned_topk_escape_prior.py --tokens outputs/stage2_semantic_tokens/20260628_stage2_semantic_tokens_32768_oi_div2k_from_stage1_rateprior0005_b16_fullroots/semantic_tokens.pt --token-prior-checkpoint checkpoints/stage2_token_prior/20260628_stage2_token_prior_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_es.pt --topk 512 --max-images 32768 --batch-size 64 --num-workers 2 --image-size 256 --wandb-mode offline --run-name 20260628_stage2_learned_topk512_escape_huffman_fit_32768tokens_stage1_rateprior0005_b16_tf384_l6_12k
```

## Summary

- tokens: outputs/stage2_semantic_tokens/20260628_stage2_semantic_tokens_32768_oi_div2k_from_stage1_rateprior0005_b16_fullroots/semantic_tokens.pt
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260628_stage2_token_prior_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_es.pt
- prior: outputs/stage2_learned_entropy/20260628_stage2_learned_topk512_escape_huffman_fit_32768tokens_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json
- num_images: 32768
- token_shape: [8, 8]
- codebook_size: 8192
- topk: 512
- event_smoothing_count: 1
- total_tokens: 2097152
- top1_hit_rate: 0.05965757369995117
- top5_hit_rate: 0.15061044692993164
- topk_hit_rate: 0.7233867645263672
- escape_rate: 0.2766132354736328
- event_entropy_bits: 6.443038058747473
- mean_unpadded_bits_per_token: 10.060627937316895
- mean_payload_bytes_per_image: 80.919921875
- mean_payload_bpp_image_size: 0.009877920150756836
- fixed_bits_payload_bpp_image_size: 0.0126953125

## Artifacts

- prior: `outputs/stage2_learned_entropy/20260628_stage2_learned_topk512_escape_huffman_fit_32768tokens_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json`
- summary: `outputs/stage2_learned_entropy/20260628_stage2_learned_topk512_escape_huffman_fit_32768tokens_stage1_rateprior0005_b16_tf384_l6_12k/summary.json`
