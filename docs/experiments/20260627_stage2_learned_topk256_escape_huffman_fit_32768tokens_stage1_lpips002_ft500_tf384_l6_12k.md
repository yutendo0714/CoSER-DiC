# 20260627_stage2_learned_topk256_escape_huffman_fit_32768tokens_stage1_lpips002_ft500_tf384_l6_12k

Date: 2026-06-27T20:04:29

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/fit_stage2_learned_topk_escape_prior.py --tokens outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_32768_oi_div2k_from_stage1_lpips002_ft500_fullroots/semantic_tokens.pt --token-prior-checkpoint checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf384_l6_12kstep_32768tokens_stage1_lpips002_ft500_lr1e4_do02_es.pt --topk 256 --max-images 32768 --batch-size 64 --num-workers 2 --wandb-mode offline --run-name 20260627_stage2_learned_topk256_escape_huffman_fit_32768tokens_stage1_lpips002_ft500_tf384_l6_12k
```

## Summary

- tokens: outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_32768_oi_div2k_from_stage1_lpips002_ft500_fullroots/semantic_tokens.pt
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf384_l6_12kstep_32768tokens_stage1_lpips002_ft500_lr1e4_do02_es.pt
- prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk256_escape_huffman_fit_32768tokens_stage1_lpips002_ft500_tf384_l6_12k/learned_topk_escape_huffman_prior.json
- num_images: 32768
- token_shape: [8, 8]
- codebook_size: 8192
- topk: 256
- event_smoothing_count: 1
- total_tokens: 2097152
- top1_hit_rate: 0.058414459228515625
- top5_hit_rate: 0.14754962921142578
- topk_hit_rate: 0.6070089340209961
- escape_rate: 0.3929910659790039
- event_entropy_bits: 5.1988810399696685
- mean_unpadded_bits_per_token: 10.360241889953613
- mean_payload_bytes_per_image: 83.31985473632812
- mean_payload_bpp_image_size: 0.01017088070511818
- fixed_bits_payload_bpp_image_size: 0.0126953125

## Artifacts

- prior: `outputs/stage2_learned_entropy/20260627_stage2_learned_topk256_escape_huffman_fit_32768tokens_stage1_lpips002_ft500_tf384_l6_12k/learned_topk_escape_huffman_prior.json`
- summary: `outputs/stage2_learned_entropy/20260627_stage2_learned_topk256_escape_huffman_fit_32768tokens_stage1_lpips002_ft500_tf384_l6_12k/summary.json`
