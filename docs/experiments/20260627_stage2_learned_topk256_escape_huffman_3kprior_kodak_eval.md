# 20260627_stage2_learned_topk256_escape_huffman_3kprior_kodak_eval

Date: 2026-06-27T10:30:09

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage2_learned_topk_escape_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --prior outputs/stage2_learned_entropy/20260627_stage2_learned_topk256_escape_huffman_fit_4096tokens_3kprior/learned_topk_escape_huffman_prior.json --image-root /dpl/kodak --max-images 24 --batch-size 4 --num-workers 4 --wandb-mode offline --run-name 20260627_stage2_learned_topk256_escape_huffman_3kprior_kodak_eval
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk256_escape_huffman_fit_4096tokens_3kprior/learned_topk_escape_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_3kstep_4096tokens_probe.pt
- num_images: 24
- crop_size: 256
- codebook_size: 8192
- topk: 256
- learned_topk_escape_huffman_payload_bpp_mean: 0.012420654296875
- learned_topk_escape_huffman_actual_bpp_mean: 0.07292683919270833
- learned_topk_escape_huffman_stream_bytes_mean: 597.4166666666666
- learned_topk_escape_huffman_payload_bytes_mean: 101.75
- learned_topk_escape_huffman_psnr_mean: 20.736252387364704
- learned_topk_escape_huffman_l1_mean: 0.06508396441737811
- learned_topk_escape_huffman_ms_ssim_mean: 0.6757684648036957
- learned_topk_escape_huffman_roundtrip_mean: 1.0
- learned_topk_escape_huffman_topk_hit_rate_mean: 0.326171875
- learned_topk_escape_huffman_unpadded_bits_per_token_mean: 12.666666666666666
- learned_topk_escape_huffman_all_tokens_roundtrip: True
- fixed_bits_payload_bpp_mean: 0.0126953125
- fixed_bits_actual_bpp_mean: 0.0731201171875
- fixed_bits_stream_bytes_mean: 599.0
- fixed_bits_payload_bytes_mean: 104.0
- fixed_bits_psnr_mean: 20.736252387364704
- fixed_bits_l1_mean: 0.06508396441737811
- fixed_bits_ms_ssim_mean: 0.6757684648036957
- fixed_bits_roundtrip_mean: 1.0
- fixed_bits_all_tokens_roundtrip: True
- learned_topk_escape_huffman_payload_bpp_delta_vs_fixed_bits: -0.000274658203125
- learned_topk_escape_huffman_payload_bpp_delta_vs_leftctx_static_huffman: 0.0006917317708333339

## Artifacts

- summary: `results/bitstreams/stage2_learned_topk_escape/20260627_stage2_learned_topk256_escape_huffman_3kprior_kodak_eval/summary.json`
- sample: `results/bitstreams/stage2_learned_topk_escape/20260627_stage2_learned_topk256_escape_huffman_3kprior_kodak_eval/stage2_learned_topk_escape_recon_grid.png`
