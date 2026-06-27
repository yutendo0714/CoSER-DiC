# 20260627_stage3_residual_ae_d32_c3_b5_r025_1000step_rate003_from2500_kodak_eval

Date: 2026-06-27T12:17:05

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage3_residual_autoencoder_bitstream.py --checkpoint checkpoints/stage3_residual_autoencoder/20260627_stage3_residual_ae_d32_c3_b5_r025_1000step_rate003_from2500_fp32.pt --semantic-prior outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json --fit-prior-root /dpl/open-images-v6/train/data --fit-prior-root /dpl/div2k --fit-prior-max-images 1024 --image-root /dpl/kodak --max-images 24 --batch-size 4 --num-workers 4 --wandb-mode offline --run-name 20260627_stage3_residual_ae_d32_c3_b5_r025_1000step_rate003_from2500_kodak_eval
```

## Summary

- checkpoint: checkpoints/stage3_residual_autoencoder/20260627_stage3_residual_ae_d32_c3_b5_r025_1000step_rate003_from2500_fp32.pt
- stage1_checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- semantic_prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json
- detail_prior: results/bitstreams/stage3_residual_autoencoder/20260627_stage3_residual_ae_d32_c3_b5_r025_1000step_rate003_from2500_kodak_eval/learned_residual_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt
- num_images: 24
- crop_size: 256
- semantic_topk: 512
- detail_shape: [3, 8, 8]
- detail_bits: 5
- detail_range: 0.25
- detail_codec: learned_residual_ae_static_huffman
- residual_code: {'codec': 'static_residual_grid_huffman', 'version': 0, 'bits': 5, 'value_range': 0.25, 'payload_codec': 'huffman', 'huffman': {'codec': 'static_huffman', 'version': 0, 'codebook_size': 32, 'code_lengths': [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1, 2, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6]}}
- fit_num_images: 1024
- fit_total_symbols: 196608
- fit_counts: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 101378, 95230, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
- fit_symbol_entropy_bits: 0.9992945194244385
- fit_mean_huffman_bits_per_symbol: 1.4843648274739583
- fit_h_detail_abs_mean: 0.0004040395982277308
- fit_h_detail_std: 0.0005131863580961249
- semantic_payload_bpp_mean: 0.010721842447916666
- detail_payload_bpp_mean: 0.004343668619791667
- total_payload_bpp_mean: 0.015065511067708334
- semantic_only_full_stream_bpp_mean: 0.07019551595052083
- stage3_full_stream_bpp_mean: 0.0751495361328125
- semantic_payload_bytes_mean: 87.83333333333333
- detail_payload_bytes_mean: 35.583333333333336
- stage3_stream_bytes_mean: 615.625
- semantic_only_psnr_mean: 20.736252387364704
- semantic_only_l1_mean: 0.06508396441737811
- semantic_only_ms_ssim_mean: 0.6757684648036957
- stage3_psnr_mean: 20.747451384862263
- stage3_l1_mean: 0.06493950650716822
- stage3_ms_ssim_mean: 0.6758395209908485
- semantic_topk_hit_rate_mean: 0.6217447916666666
- semantic_token_roundtrip_mean: 1.0
- detail_code_roundtrip_mean: 1.0
- h_detail_abs_mean_mean: 0.0003933383134911613
- h_detail_std_mean: 0.0004753764575677148
- detail_code_entropy_bits_mean: 0.9928401882449785
- stage3_psnr_delta_vs_semantic_only: 0.011198997497558594
- stage3_l1_delta_vs_semantic_only: -0.00014445791020989418
- stage3_ms_ssim_delta_vs_semantic_only: 7.105618715286255e-05
- all_semantic_tokens_roundtrip: True
- all_detail_codes_roundtrip: True
- roundtrip_failure_count: 0
- roundtrip_failures: []

## Artifacts

- summary: `results/bitstreams/stage3_residual_autoencoder/20260627_stage3_residual_ae_d32_c3_b5_r025_1000step_rate003_from2500_kodak_eval/summary.json`
- sample: `results/bitstreams/stage3_residual_autoencoder/20260627_stage3_residual_ae_d32_c3_b5_r025_1000step_rate003_from2500_kodak_eval/stage3_residual_autoencoder.png`
- detail_prior: `results/bitstreams/stage3_residual_autoencoder/20260627_stage3_residual_ae_d32_c3_b5_r025_1000step_rate003_from2500_kodak_eval/learned_residual_huffman_prior.json`
