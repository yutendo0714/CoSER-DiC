# 20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_stage1_lpips002_ft500_lowlr8k

Date: 2026-06-27T19:09:49

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/fit_stage2_learned_topk_escape_prior.py --tokens outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_32768_oi_div2k_from_stage1_lpips002_ft500_fullroots/semantic_tokens.pt --token-prior-checkpoint checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_stage1_lpips002_ft500_lr1e4_do02_es.pt --topk 512 --max-images 32768 --batch-size 64 --num-workers 2 --wandb-mode offline --run-name 20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_stage1_lpips002_ft500_lowlr8k
```

## Summary

- tokens: outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_32768_oi_div2k_from_stage1_lpips002_ft500_fullroots/semantic_tokens.pt
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_stage1_lpips002_ft500_lr1e4_do02_es.pt
- prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_stage1_lpips002_ft500_lowlr8k/learned_topk_escape_huffman_prior.json
- num_images: 32768
- token_shape: [8, 8]
- codebook_size: 8192
- topk: 512
- event_smoothing_count: 1
- total_tokens: 2097152
- top1_hit_rate: 0.0462646484375
- top5_hit_rate: 0.11911582946777344
- topk_hit_rate: 0.6523785591125488
- escape_rate: 0.34762144088745117
- event_entropy_bits: 6.106121967679597
- mean_unpadded_bits_per_token: 10.676669597625732
- mean_payload_bytes_per_image: 85.85037231445312
- mean_payload_bpp_image_size: 0.010479781776666641
- fixed_bits_payload_bpp_image_size: 0.0126953125

## Artifacts

- prior: `outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_stage1_lpips002_ft500_lowlr8k/learned_topk_escape_huffman_prior.json`
- summary: `outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_stage1_lpips002_ft500_lowlr8k/summary.json`
