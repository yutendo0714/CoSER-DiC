# 20260627_stage2_learned_topk512_escape_huffman_fit_4096tokens_stage1_rateprior0005_tau01_vq005_ft500_lowlr3k

Date: 2026-06-27T22:38:07

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/fit_stage2_learned_topk_escape_prior.py --tokens outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_rateprior0005_tau01_vq005_ft500_fullroots/semantic_tokens.pt --token-prior-checkpoint checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_3kstep_4096tokens_stage1_rateprior0005_tau01_vq005_ft500_lr1e4_do02_es.pt --topk 512 --batch-size 64 --num-workers 2 --wandb-mode offline --run-name 20260627_stage2_learned_topk512_escape_huffman_fit_4096tokens_stage1_rateprior0005_tau01_vq005_ft500_lowlr3k
```

## Summary

- tokens: outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_rateprior0005_tau01_vq005_ft500_fullroots/semantic_tokens.pt
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_3kstep_4096tokens_stage1_rateprior0005_tau01_vq005_ft500_lr1e4_do02_es.pt
- prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_4096tokens_stage1_rateprior0005_tau01_vq005_ft500_lowlr3k/learned_topk_escape_huffman_prior.json
- num_images: 4096
- token_shape: [8, 8]
- codebook_size: 8192
- topk: 512
- event_smoothing_count: 1
- total_tokens: 262144
- top1_hit_rate: 0.06806182861328125
- top5_hit_rate: 0.1661529541015625
- topk_hit_rate: 0.7982139587402344
- escape_rate: 0.20178604125976562
- event_entropy_bits: 6.863686337934563
- mean_unpadded_bits_per_token: 9.517200469970703
- mean_payload_bytes_per_image: 76.576904296875
- mean_payload_bpp_image_size: 0.009347766637802124
- fixed_bits_payload_bpp_image_size: 0.0126953125

## Artifacts

- prior: `outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_4096tokens_stage1_rateprior0005_tau01_vq005_ft500_lowlr3k/learned_topk_escape_huffman_prior.json`
- summary: `outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_4096tokens_stage1_rateprior0005_tau01_vq005_ft500_lowlr3k/summary.json`
