# 20260627_stage3_residual_ae_d32_c3_b5_r025_2500step_norate_kodak_eval

Date: 2026-06-27T12:14:14

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage3_residual_autoencoder_bitstream.py --checkpoint checkpoints/stage3_residual_autoencoder/20260627_stage3_residual_ae_d32_c3_b5_r025_2500step_norate_from500_fp32.pt --semantic-prior outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json --fit-prior-root /dpl/open-images-v6/train/data --fit-prior-root /dpl/div2k --fit-prior-max-images 1024 --image-root /dpl/kodak --max-images 24 --batch-size 4 --num-workers 4 --wandb-mode offline --run-name 20260627_stage3_residual_ae_d32_c3_b5_r025_2500step_norate_kodak_eval
```

## Summary

- checkpoint: checkpoints/stage3_residual_autoencoder/20260627_stage3_residual_ae_d32_c3_b5_r025_2500step_norate_from500_fp32.pt
- stage1_checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- semantic_prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json
- detail_prior: results/bitstreams/stage3_residual_autoencoder/20260627_stage3_residual_ae_d32_c3_b5_r025_2500step_norate_kodak_eval/learned_residual_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt
- num_images: 24
- crop_size: 256
- semantic_topk: 512
- detail_shape: [3, 8, 8]
- detail_bits: 5
- detail_range: 0.25
- detail_codec: learned_residual_ae_static_huffman
- residual_code: {'codec': 'static_residual_grid_huffman', 'version': 0, 'bits': 5, 'value_range': 0.25, 'payload_codec': 'huffman', 'huffman': {'codec': 'static_huffman', 'version': 0, 'codebook_size': 32, 'code_lengths': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]}}
- fit_num_images: 1024
- fit_total_symbols: 196608
- fit_counts: [5884, 7877, 6814, 6417, 6228, 5940, 5834, 5810, 5720, 5480, 5443, 5647, 5577, 5368, 5632, 5543, 5491, 5642, 5552, 5593, 5609, 5705, 5610, 5862, 5820, 5984, 6281, 6457, 7093, 7502, 9496, 7697]
- fit_symbol_entropy_bits: 4.985955238342285
- fit_mean_huffman_bits_per_symbol: 5.0
- fit_h_detail_abs_mean: 0.13680958359812698
- fit_h_detail_std: 0.15618659677371197
- semantic_payload_bpp_mean: 0.010721842447916666
- detail_payload_bpp_mean: 0.0146484375
- total_payload_bpp_mean: 0.025370279947916668
- semantic_only_full_stream_bpp_mean: 0.07019551595052083
- stage3_full_stream_bpp_mean: 0.08557637532552083
- semantic_payload_bytes_mean: 87.83333333333333
- detail_payload_bytes_mean: 120.0
- stage3_stream_bytes_mean: 701.0416666666666
- semantic_only_psnr_mean: 20.736252387364704
- semantic_only_l1_mean: 0.06508396441737811
- semantic_only_ms_ssim_mean: 0.6757684648036957
- stage3_psnr_mean: 20.94353238741557
- stage3_l1_mean: 0.06226776074618101
- stage3_ms_ssim_mean: 0.6873576069871584
- semantic_topk_hit_rate_mean: 0.6217447916666666
- semantic_token_roundtrip_mean: 1.0
- detail_code_roundtrip_mean: 1.0
- h_detail_abs_mean_mean: 0.13273724789420763
- h_detail_std_mean: 0.14956466543177763
- detail_code_entropy_bits_mean: 4.841707011063893
- stage3_psnr_delta_vs_semantic_only: 0.207280000050865
- stage3_l1_delta_vs_semantic_only: -0.002816203671197101
- stage3_ms_ssim_delta_vs_semantic_only: 0.011589142183462742
- all_semantic_tokens_roundtrip: True
- all_detail_codes_roundtrip: True
- roundtrip_failure_count: 0
- roundtrip_failures: []

## Artifacts

- summary: `results/bitstreams/stage3_residual_autoencoder/20260627_stage3_residual_ae_d32_c3_b5_r025_2500step_norate_kodak_eval/summary.json`
- sample: `results/bitstreams/stage3_residual_autoencoder/20260627_stage3_residual_ae_d32_c3_b5_r025_2500step_norate_kodak_eval/stage3_residual_autoencoder.png`
- detail_prior: `results/bitstreams/stage3_residual_autoencoder/20260627_stage3_residual_ae_d32_c3_b5_r025_2500step_norate_kodak_eval/learned_residual_huffman_prior.json`
