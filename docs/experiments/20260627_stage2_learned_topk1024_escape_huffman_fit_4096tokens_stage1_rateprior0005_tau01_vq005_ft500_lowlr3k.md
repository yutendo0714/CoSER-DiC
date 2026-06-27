# 20260627_stage2_learned_topk1024_escape_huffman_fit_4096tokens_stage1_rateprior0005_tau01_vq005_ft500_lowlr3k

Date: 2026-06-27T22:58:21

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/fit_stage2_learned_topk_escape_prior.py --tokens outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_rateprior0005_tau01_vq005_ft500_fullroots/semantic_tokens.pt --token-prior-checkpoint checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_3kstep_4096tokens_stage1_rateprior0005_tau01_vq005_ft500_lr1e4_do02_es.pt --topk 1024 --batch-size 64 --num-workers 2 --wandb-mode offline --run-name 20260627_stage2_learned_topk1024_escape_huffman_fit_4096tokens_stage1_rateprior0005_tau01_vq005_ft500_lowlr3k
```

## Summary

- tokens: outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_rateprior0005_tau01_vq005_ft500_fullroots/semantic_tokens.pt
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_3kstep_4096tokens_stage1_rateprior0005_tau01_vq005_ft500_lr1e4_do02_es.pt
- prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk1024_escape_huffman_fit_4096tokens_stage1_rateprior0005_tau01_vq005_ft500_lowlr3k/learned_topk_escape_huffman_prior.json
- num_images: 4096
- token_shape: [8, 8]
- codebook_size: 8192
- topk: 1024
- event_smoothing_count: 1
- total_tokens: 262144
- top1_hit_rate: 0.06806182861328125
- top5_hit_rate: 0.1661529541015625
- topk_hit_rate: 0.8892364501953125
- escape_rate: 0.1107635498046875
- event_entropy_bits: 7.876585942849649
- mean_unpadded_bits_per_token: 9.34146499633789
- mean_payload_bytes_per_image: 75.17333984375
- mean_payload_bpp_image_size: 0.009176433086395264
- fixed_bits_payload_bpp_image_size: 0.0126953125

## Artifacts

- prior: `outputs/stage2_learned_entropy/20260627_stage2_learned_topk1024_escape_huffman_fit_4096tokens_stage1_rateprior0005_tau01_vq005_ft500_lowlr3k/learned_topk_escape_huffman_prior.json`
- summary: `outputs/stage2_learned_entropy/20260627_stage2_learned_topk1024_escape_huffman_fit_4096tokens_stage1_rateprior0005_tau01_vq005_ft500_lowlr3k/summary.json`
