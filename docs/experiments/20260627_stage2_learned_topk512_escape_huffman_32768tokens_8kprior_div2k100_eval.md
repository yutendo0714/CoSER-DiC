# 20260627_stage2_learned_topk512_escape_huffman_32768tokens_8kprior_div2k100_eval

Date: 2026-06-27T11:20:31

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage2_learned_topk_escape_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --prior outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json --image-root /dpl/div2k --max-images 100 --batch-size 4 --num-workers 4 --wandb-mode offline --run-name 20260627_stage2_learned_topk512_escape_huffman_32768tokens_8kprior_div2k100_eval
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt
- num_images: 100
- crop_size: 256
- codebook_size: 8192
- topk: 512
- learned_topk_escape_huffman_payload_bpp_mean: 0.0104248046875
- learned_topk_escape_huffman_actual_bpp_mean: 0.070870361328125
- learned_topk_escape_huffman_stream_bytes_mean: 580.57
- learned_topk_escape_huffman_payload_bytes_mean: 85.4
- learned_topk_escape_huffman_psnr_mean: 20.729811868667603
- learned_topk_escape_huffman_l1_mean: 0.06994628557004035
- learned_topk_escape_huffman_ms_ssim_mean: 0.6833488121628761
- learned_topk_escape_huffman_roundtrip_mean: 0.99
- learned_topk_escape_huffman_topk_hit_rate_mean: 0.64609375
- learned_topk_escape_huffman_unpadded_bits_per_token_mean: 10.6190625
- learned_topk_escape_huffman_all_tokens_roundtrip: False
- fixed_bits_payload_bpp_mean: 0.0126953125
- fixed_bits_actual_bpp_mean: 0.0731201171875
- fixed_bits_stream_bytes_mean: 599.0
- fixed_bits_payload_bytes_mean: 104.0
- fixed_bits_psnr_mean: 20.736100835800173
- fixed_bits_l1_mean: 0.0698748239222914
- fixed_bits_ms_ssim_mean: 0.6836348041892052
- fixed_bits_roundtrip_mean: 1.0
- fixed_bits_all_tokens_roundtrip: True
- learned_topk_escape_huffman_payload_bpp_delta_vs_fixed_bits: -0.0022705078124999993
- learned_topk_escape_huffman_payload_bpp_delta_vs_leftctx_static_huffman: -0.0013041178385416654

## Artifacts

- summary: `results/bitstreams/stage2_learned_topk_escape/20260627_stage2_learned_topk512_escape_huffman_32768tokens_8kprior_div2k100_eval/summary.json`
- sample: `results/bitstreams/stage2_learned_topk_escape/20260627_stage2_learned_topk512_escape_huffman_32768tokens_8kprior_div2k100_eval/stage2_learned_topk_escape_recon_grid.png`
