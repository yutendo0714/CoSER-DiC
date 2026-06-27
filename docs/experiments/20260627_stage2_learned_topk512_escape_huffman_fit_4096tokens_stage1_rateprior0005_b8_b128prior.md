# 20260627_stage2_learned_topk512_escape_huffman_fit_4096tokens_stage1_rateprior0005_b8_b128prior

Date: 2026-06-27T23:55:39

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/fit_stage2_learned_topk_escape_prior.py --tokens outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_rateprior0005_b8_fullroots/semantic_tokens.pt --token-prior-checkpoint checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_3kstep_4096tokens_stage1_rateprior0005_b8_lr1e4_do02_b128_es.pt --topk 512 --event-smoothing-count 1 --batch-size 128 --num-workers 2 --image-size 256 --wandb-mode offline --run-name 20260627_stage2_learned_topk512_escape_huffman_fit_4096tokens_stage1_rateprior0005_b8_b128prior
```

## Summary

- tokens: outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_rateprior0005_b8_fullroots/semantic_tokens.pt
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_3kstep_4096tokens_stage1_rateprior0005_b8_lr1e4_do02_b128_es.pt
- prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_4096tokens_stage1_rateprior0005_b8_b128prior/learned_topk_escape_huffman_prior.json
- num_images: 4096
- token_shape: [8, 8]
- codebook_size: 8192
- topk: 512
- event_smoothing_count: 1
- total_tokens: 262144
- top1_hit_rate: 0.07954788208007812
- top5_hit_rate: 0.1920318603515625
- topk_hit_rate: 0.8288307189941406
- escape_rate: 0.17116928100585938
- event_entropy_bits: 6.874795805470406
- mean_unpadded_bits_per_token: 9.141033172607422
- mean_payload_bytes_per_image: 73.5546875
- mean_payload_bpp_image_size: 0.008978843688964844
- fixed_bits_payload_bpp_image_size: 0.0126953125

## Artifacts

- prior: `outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_4096tokens_stage1_rateprior0005_b8_b128prior/learned_topk_escape_huffman_prior.json`
- summary: `outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_4096tokens_stage1_rateprior0005_b8_b128prior/summary.json`
