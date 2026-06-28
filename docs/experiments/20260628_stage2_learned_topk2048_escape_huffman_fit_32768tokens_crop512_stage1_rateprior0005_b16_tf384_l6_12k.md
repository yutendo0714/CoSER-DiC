# 20260628_stage2_learned_topk2048_escape_huffman_fit_32768tokens_crop512_stage1_rateprior0005_b16_tf384_l6_12k

Date: 2026-06-28T04:13:47

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/fit_stage2_learned_topk_escape_prior.py --tokens outputs/stage2_semantic_tokens/20260628_stage2_semantic_tokens_32768_oi_val_crop512_stage1_rateprior0005_b16/semantic_tokens.pt --token-prior-checkpoint checkpoints/stage2_token_prior/20260628_stage2_token_prior_crop512_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_amp_es.pt --topk 2048 --event-smoothing-count 1 --batch-size 64 --num-workers 4 --image-size 512 --wandb-mode offline --run-name 20260628_stage2_learned_topk2048_escape_huffman_fit_32768tokens_crop512_stage1_rateprior0005_b16_tf384_l6_12k
```

## Summary

- tokens: outputs/stage2_semantic_tokens/20260628_stage2_semantic_tokens_32768_oi_val_crop512_stage1_rateprior0005_b16/semantic_tokens.pt
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260628_stage2_token_prior_crop512_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_amp_es.pt
- prior: outputs/stage2_learned_entropy/20260628_stage2_learned_topk2048_escape_huffman_fit_32768tokens_crop512_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json
- num_images: 32768
- token_shape: [16, 16]
- codebook_size: 8192
- topk: 2048
- event_smoothing_count: 1
- total_tokens: 8388608
- top1_hit_rate: 0.0960930585861206
- top5_hit_rate: 0.21870803833007812
- topk_hit_rate: 0.9494014978408813
- escape_rate: 0.05059850215911865
- event_entropy_bits: 8.278229300690224
- mean_unpadded_bits_per_token: 8.97101354598999
- mean_payload_bytes_per_image: 287.5087585449219
- mean_payload_bpp_image_size: 0.008774071000516415
- fixed_bits_payload_bpp_image_size: 0.0126953125

## Artifacts

- prior: `outputs/stage2_learned_entropy/20260628_stage2_learned_topk2048_escape_huffman_fit_32768tokens_crop512_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json`
- summary: `outputs/stage2_learned_entropy/20260628_stage2_learned_topk2048_escape_huffman_fit_32768tokens_crop512_stage1_rateprior0005_b16_tf384_l6_12k/summary.json`
