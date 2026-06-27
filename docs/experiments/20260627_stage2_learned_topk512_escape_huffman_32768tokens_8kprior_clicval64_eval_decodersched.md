# 20260627_stage2_learned_topk512_escape_huffman_32768tokens_8kprior_clicval64_eval_decodersched

Date: 2026-06-27T11:23:46

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage2_learned_topk_escape_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --prior outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json --image-root /dpl/clic/professional/valid --max-images 64 --batch-size 4 --num-workers 4 --wandb-mode offline --run-name 20260627_stage2_learned_topk512_escape_huffman_32768tokens_8kprior_clicval64_eval_decodersched
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt
- num_images: 41
- crop_size: 256
- codebook_size: 8192
- topk: 512
- learned_topk_escape_huffman_payload_bpp_mean: 0.009905559260670731
- learned_topk_escape_huffman_actual_bpp_mean: 0.0703392959222561
- learned_topk_escape_huffman_stream_bytes_mean: 576.219512195122
- learned_topk_escape_huffman_payload_bytes_mean: 81.14634146341463
- learned_topk_escape_huffman_psnr_mean: 22.469522429675592
- learned_topk_escape_huffman_l1_mean: 0.05876288369934007
- learned_topk_escape_huffman_ms_ssim_mean: 0.7211726809420237
- learned_topk_escape_huffman_roundtrip_mean: 1.0
- learned_topk_escape_huffman_topk_hit_rate_mean: 0.7164634146341463
- learned_topk_escape_huffman_unpadded_bits_per_token_mean: 10.091082317073171
- learned_topk_escape_huffman_all_tokens_roundtrip: True
- learned_topk_escape_huffman_roundtrip_failure_count: 0
- fixed_bits_payload_bpp_mean: 0.0126953125
- fixed_bits_actual_bpp_mean: 0.0731201171875
- fixed_bits_stream_bytes_mean: 599.0
- fixed_bits_payload_bytes_mean: 104.0
- fixed_bits_psnr_mean: 22.469522429675592
- fixed_bits_l1_mean: 0.05876288369934007
- fixed_bits_ms_ssim_mean: 0.7211726809420237
- fixed_bits_roundtrip_mean: 1.0
- fixed_bits_all_tokens_roundtrip: True
- fixed_bits_roundtrip_failure_count: 0
- learned_topk_escape_huffman_payload_bpp_delta_vs_fixed_bits: -0.002789753239329269
- learned_topk_escape_huffman_payload_bpp_delta_vs_leftctx_static_huffman: -0.001823363265370935
- roundtrip_failures: []

## Artifacts

- summary: `results/bitstreams/stage2_learned_topk_escape/20260627_stage2_learned_topk512_escape_huffman_32768tokens_8kprior_clicval64_eval_decodersched/summary.json`
- sample: `results/bitstreams/stage2_learned_topk_escape/20260627_stage2_learned_topk512_escape_huffman_32768tokens_8kprior_clicval64_eval_decodersched/stage2_learned_topk_escape_recon_grid.png`
