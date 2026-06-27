# 20260627_stage3_residual_ae_d32_c3_b5_r025_500step_norate_kodak_eval

Date: 2026-06-27T12:07:23

## Command

```bash
/workspace/CoSER-DiC/.venv/bin/python scripts/eval_stage3_residual_autoencoder_bitstream.py --checkpoint checkpoints/stage3_residual_autoencoder/20260627_stage3_residual_ae_d32_c3_b5_r025_500step_norate_fp32.pt --semantic-prior outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json --fit-prior-root /dpl/open-images-v6/train/data --fit-prior-root /dpl/div2k --fit-prior-max-images 1024 --image-root /dpl/kodak --max-images 24 --batch-size 4 --num-workers 4 --wandb-mode offline --run-name 20260627_stage3_residual_ae_d32_c3_b5_r025_500step_norate_kodak_eval
```

## Summary

- checkpoint: checkpoints/stage3_residual_autoencoder/20260627_stage3_residual_ae_d32_c3_b5_r025_500step_norate_fp32.pt
- stage1_checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
- semantic_prior: outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json
- detail_prior: results/bitstreams/stage3_residual_autoencoder/20260627_stage3_residual_ae_d32_c3_b5_r025_500step_norate_kodak_eval/learned_residual_huffman_prior.json
- token_prior_checkpoint: checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt
- num_images: 24
- crop_size: 256
- semantic_topk: 512
- detail_shape: [3, 8, 8]
- detail_bits: 5
- detail_range: 0.25
- detail_codec: learned_residual_ae_static_huffman
- residual_code: {'codec': 'static_residual_grid_huffman', 'version': 0, 'bits': 5, 'value_range': 0.25, 'payload_codec': 'huffman', 'huffman': {'codec': 'static_huffman', 'version': 0, 'codebook_size': 32, 'code_lengths': [20, 20, 19, 19, 19, 19, 19, 15, 13, 11, 9, 7, 5, 4, 3, 1, 2, 6, 8, 10, 12, 14, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19]}}
- fit_num_images: 1024
- fit_total_symbols: 196608
- fit_counts: [0, 0, 0, 0, 0, 0, 0, 9, 70, 387, 1152, 2713, 6361, 14035, 31123, 93608, 40155, 5138, 1303, 410, 125, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
- fit_symbol_entropy_bits: 2.193880081176758
- fit_mean_huffman_bits_per_symbol: 2.222752888997396
- fit_h_detail_abs_mean: 0.015053404485418772
- fit_h_detail_std: 0.02263500632581726
- semantic_payload_bpp_mean: 0.010721842447916666
- detail_payload_bpp_mean: 0.0065765380859375
- total_payload_bpp_mean: 0.017298380533854168
- semantic_only_full_stream_bpp_mean: 0.07019551595052083
- stage3_full_stream_bpp_mean: 0.07738240559895833
- semantic_payload_bytes_mean: 87.83333333333333
- detail_payload_bytes_mean: 53.875
- stage3_stream_bytes_mean: 633.9166666666666
- semantic_only_psnr_mean: 20.736252387364704
- semantic_only_l1_mean: 0.06508396441737811
- semantic_only_ms_ssim_mean: 0.6757684648036957
- stage3_psnr_mean: 20.745051622390747
- stage3_l1_mean: 0.06496650857540469
- stage3_ms_ssim_mean: 0.6758460154136022
- semantic_topk_hit_rate_mean: 0.6217447916666666
- semantic_token_roundtrip_mean: 1.0
- detail_code_roundtrip_mean: 1.0
- h_detail_abs_mean_mean: 0.014694066563000282
- h_detail_std_mean: 0.019323961227200925
- detail_code_entropy_bits_mean: 2.0815958778063455
- stage3_psnr_delta_vs_semantic_only: 0.008799235026042851
- stage3_l1_delta_vs_semantic_only: -0.00011745584197342396
- stage3_ms_ssim_delta_vs_semantic_only: 7.755060990655149e-05
- all_semantic_tokens_roundtrip: True
- all_detail_codes_roundtrip: True
- roundtrip_failure_count: 0
- roundtrip_failures: []

## Artifacts

- summary: `results/bitstreams/stage3_residual_autoencoder/20260627_stage3_residual_ae_d32_c3_b5_r025_500step_norate_kodak_eval/summary.json`
- sample: `results/bitstreams/stage3_residual_autoencoder/20260627_stage3_residual_ae_d32_c3_b5_r025_500step_norate_kodak_eval/stage3_residual_autoencoder.png`
- detail_prior: `results/bitstreams/stage3_residual_autoencoder/20260627_stage3_residual_ae_d32_c3_b5_r025_500step_norate_kodak_eval/learned_residual_huffman_prior.json`
