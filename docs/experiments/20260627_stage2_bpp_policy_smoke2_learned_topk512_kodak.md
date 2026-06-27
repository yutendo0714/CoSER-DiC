# 20260627_stage2_bpp_policy_smoke2_learned_topk512_kodak

Date: 2026-06-27T15:04:02

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage2_learned_topk_escape_bitstream.py --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt --prior outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json --token-prior-checkpoint checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt --image-root /dpl/kodak --crop-size 256 --max-images 2 --batch-size 2 --num-workers 2 --stream-header-codec compact --stream-checksum-codec crc32 --wandb-mode offline --run-name 20260627_stage2_bpp_policy_smoke2_learned_topk512_kodak
```

## Summary

- checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt
- num_images: 2
- crop_size: 256
- codebook_size: 8192
- topk: 512
- stream_header_codec: compact
- stream_checksum_codec: crc32
- main_bpp_metric: learned_topk_escape_huffman_actual_payload_bpp_mean
- paper_bpp_metric: learned_topk_escape_huffman_paper_bpp_mean
- debug_bpp_metric: learned_topk_escape_huffman_debug_full_stream_bpp_mean
- learned_topk_escape_huffman_payload_bpp_mean: 0.00982666015625
- learned_topk_escape_huffman_actual_payload_bpp_mean: 0.00982666015625
- learned_topk_escape_huffman_paper_bpp_mean: 0.00982666015625
- learned_topk_escape_huffman_actual_bpp_mean: 0.01458740234375
- learned_topk_escape_huffman_debug_full_stream_bpp_mean: 0.01458740234375
- learned_topk_escape_huffman_stream_bytes_mean: 119.5
- learned_topk_escape_huffman_payload_bytes_mean: 80.5
- learned_topk_escape_huffman_psnr_mean: 21.331958770751953
- learned_topk_escape_huffman_l1_mean: 0.061721380800008774
- learned_topk_escape_huffman_ms_ssim_mean: 0.6254555284976959
- learned_topk_escape_huffman_roundtrip_mean: 1.0
- learned_topk_escape_huffman_topk_hit_rate_mean: 0.75
- learned_topk_escape_huffman_unpadded_bits_per_token_mean: 10.046875
- learned_topk_escape_huffman_all_tokens_roundtrip: True
- learned_topk_escape_huffman_roundtrip_failure_count: 0
- fixed_bits_payload_bpp_mean: 0.0126953125
- fixed_bits_actual_payload_bpp_mean: 0.0126953125
- fixed_bits_paper_bpp_mean: 0.0126953125
- fixed_bits_actual_bpp_mean: 0.0174560546875
- fixed_bits_debug_full_stream_bpp_mean: 0.0174560546875
- fixed_bits_stream_bytes_mean: 143.0
- fixed_bits_payload_bytes_mean: 104.0
- fixed_bits_psnr_mean: 21.331958770751953
- fixed_bits_l1_mean: 0.061721380800008774
- fixed_bits_ms_ssim_mean: 0.6254555284976959
- fixed_bits_roundtrip_mean: 1.0
- fixed_bits_all_tokens_roundtrip: True
- fixed_bits_roundtrip_failure_count: 0
- learned_topk_escape_huffman_payload_bpp_delta_vs_fixed_bits: -0.00286865234375
- learned_topk_escape_huffman_payload_bpp_delta_vs_leftctx_static_huffman: -0.001902262369791666
- roundtrip_failures: []

## Artifacts

- summary: `results/bitstreams/stage2_learned_topk_escape/20260627_stage2_bpp_policy_smoke2_learned_topk512_kodak/summary.json`
- sample: `results/bitstreams/stage2_learned_topk_escape/20260627_stage2_bpp_policy_smoke2_learned_topk512_kodak/stage2_learned_topk_escape_recon_grid.png`
