# Stage 2 Semantic Entropy Progress

Date: 2026-06-27 JST

## Starting Point

The best Stage 1 semantic VQ checkpoint is:

```text
checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
```

Stage 2 initially used global static Huffman fit from 256 calibration images.
Kodak payload bpp improved from fixed-width 0.012695 to 0.012054.

## Implemented Today

Added:

```text
StaticLeftContextHuffmanCode
StaticANSCode
--prior-kind left_context
--prior-kind ans
compact fixed-length entropy_model_version IDs in the Stage 2 eval container
```

The compact header IDs remove a debug-container artifact where longer codec
names inflated full-stream bpp.

## Key Experiments

4096-image global static Huffman:

```text
fit:
  20260627_stage2_static_huffman_prior_4096calib_oi_div2k_from_stage1_best
  wandb/offline-run-20260627_100002-dkj3ec9s

eval:
  20260627_stage2_static_huffman_4096calib_oi_div2k_kodak_eval_compacthdr
  wandb/offline-run-20260627_100216-ktw8unui

Kodak:
  payload_bpp: 0.011820
  full_stream_bpp: 0.071533
  payload_bytes/image: 96.83
  token_roundtrip: true
```

4096-image left-context static Huffman:

```text
fit:
  20260627_stage2_leftctx_huffman_top64_b4096_4096calib_oi_div2k_from_stage1_best
  wandb/offline-run-20260627_100057-9ry5exbw

eval:
  20260627_stage2_leftctx_huffman_top64_b4096_4096calib_oi_div2k_kodak_eval_compacthdr
  wandb/offline-run-20260627_100231-92tm3m06

Kodak:
  payload_bpp: 0.011729
  full_stream_bpp: 0.071437
  payload_bytes/image: 96.08
  token_roundtrip: true
```

4096-image static ANS:

```text
fit:
  20260627_stage2_static_ans_prior_4096calib_oi_div2k_from_stage1_best
  wandb/offline-run-20260627_100550-vr54e2zr

eval:
  20260627_stage2_static_ans_4096calib_oi_div2k_kodak_eval_compacthdr
  wandb/offline-run-20260627_100606-1w8hj69p

Kodak:
  payload_bpp: 0.012451
  full_stream_bpp: 0.072255
  payload_bytes/image: 102.00
  token_roundtrip: true
```

## Static-Prior Decision

At this point in the day, the active Stage 2 semantic-token prior was:

```text
4096-calibrated left-context static Huffman
```

Reason:

```text
It is the best actual transmitted payload/full-stream result so far while
preserving exact token roundtrip and decoder-known causal context.
```

Static ANS remains useful infrastructure for future decoder-known CDF probes,
but it is not active for the current 64-token per-image semantic stream because
per-image flush overhead dominates.

## Next Research Step

Moved from handcrafted static context to a CoSER-owned learned token-prior
probe, while keeping actual transmitted bpp as the adoption criterion:

```text
export:
  20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_best
  wandb/offline-run-20260627_100942-jf2j05ir
  outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_best/semantic_tokens.pt

token count:
  num_images: 4096
  token_shape: 8x8
  total_tokens: 262144
  active_codes: 7520 / 8192
  global_entropy_bits: 11.9197
```

Implemented:

```text
src/coserdic/models/token_prior.py
scripts/export_stage1_semantic_tokens.py
scripts/train_stage2_token_prior.py
tests/test_token_prior.py
```

Best learned-prior probe so far:

```text
20260627_stage2_token_prior_tf256_l4_3kstep_4096tokens_probe
wandb/offline-run-20260627_101223-ox76o7pf

best val cross-entropy: 11.2619 bits/token
final val cross-entropy: 11.2871 bits/token
final val top1 / top5 / top64: 0.0310 / 0.0777 / 0.2231
```

Longer 10k and 5k probes overfit or destabilized after the early best point,
even after fixing AMP gradient clipping order:

```text
10k probe best: 11.2796 bits/token, final: 12.2071 bits/token
5k unscale/clip probe best: 11.2728 bits/token, final: 18.9196 bits/token
```

## Initial Learned-Prior Decision

The 4096-token learned prior was promising because its held-out estimated
bits/token were below the 4096-calibrated left-context static Huffman
calibration average:

```text
learned prior best val estimate: 11.2619 bits/token
left-context static Huffman calibration: 11.7991 bits/token
fixed_bits: 13 bits/token
```

It was not promoted at this point because this was still estimated
cross-entropy, not transmitted bytes from compress/decompress. The active
actual-bpp baseline still remained:

```text
4096-calibrated left-context static Huffman
Kodak payload_bpp: 0.011729
Kodak full_stream_bpp: 0.071437
token_roundtrip: true
```

Next research step:

```text
1. Export a larger semantic-token dataset, starting with 32768 images.
2. Train the same simple causal Transformer with lower LR, stronger dropout,
   stronger weight decay, and early stopping.
3. Implement an actual causal coding backend for learned probabilities:
   RDVQ-style top-k/escape coding or chunked rANS.
4. Only promote learned prior after actual compress/decompress payload bpp
   beats the left-context static Huffman baseline on Kodak.
```

## Actual Learned Top-k/Escape Bridge

Implemented and tested:

```text
TopKEscapeHuffmanCode
fit_stage2_learned_topk_escape_prior.py
eval_stage2_learned_topk_escape_bitstream.py
```

This is a CoSER-owned bridge inspired by RDVQ top-k/escape coding. It is not a
direct RDVQ port and not the final tensor-rANS backend. It uses decoder-rebuilt
top-k candidates and a Huffman-coded rank/escape event stream, followed by a
fixed-width token id for escapes.

4096-token calibration:

```text
top64  payload_bpp: 0.010615, topk_hit: 0.41695
top128 payload_bpp: 0.010279, topk_hit: 0.51890
top256 payload_bpp: 0.010031, topk_hit: 0.63268
```

Kodak actual compress/decompress:

```text
top64:
  run: 20260627_stage2_learned_topk64_escape_huffman_3kprior_kodak_eval
  wandb/offline-run-20260627_102741-sl976cl1
  payload_bpp: 0.012380
  full_stream_bpp: 0.072891
  topk_hit: 0.18750
  roundtrip: true

top128:
  run: 20260627_stage2_learned_topk128_escape_huffman_3kprior_kodak_eval
  wandb/offline-run-20260627_102845-7jwvagp4
  payload_bpp: 0.012161
  full_stream_bpp: 0.072662
  topk_hit: 0.24870
  roundtrip: true

top256:
  run: 20260627_stage2_learned_topk256_escape_huffman_3kprior_kodak_eval
  wandb/offline-run-20260627_103009-gyyh3v9b
  payload_bpp: 0.012421
  full_stream_bpp: 0.072927
  topk_hit: 0.32617
  roundtrip: true
```

4096-token decision:

```text
Do not adopt the 4096-token learned top-k/escape probe.

It beats fixed_bits, with top128 best among the quick sweep:
  fixed_bits payload_bpp: 0.012695
  learned top128 payload_bpp: 0.012161

But it does not beat the then-active left-context static Huffman:
  left-context payload_bpp: 0.011729
```

Next learned-prior work should focus on distribution generalization before
more elaborate coding: export more tokens, train with lower LR/regularization,
and then replace the Huffman bridge with RDVQ-style tensor-rANS if the
probabilities improve.

## 11:20 JST Update - 32768-Token Learned Top-k/Escape

Executed the planned larger-token path:

```text
export:
  20260627_stage2_semantic_tokens_32768_oi_div2k_from_stage1_best
  wandb/offline-run-20260627_104221-imbu1ilj

token prior:
  20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es
  wandb/offline-run-20260627_104246-nebvdh5a
  best/final val cross-entropy: 9.8837 bits/token
  final val top1 / top5 / top64: 0.0494 / 0.1294 / 0.3615
```

Actual-byte top-k/escape sweep on Kodak:

```text
top64:
  fit wandb: wandb/offline-run-20260627_105153-n922o2xj
  eval wandb: wandb/offline-run-20260627_105211-l9p5fys4
  payload_bpp: 0.011383
  full_stream_bpp: 0.071849
  topk_hit: 0.31510
  roundtrip: true

top128:
  fit wandb: wandb/offline-run-20260627_104902-nbwamsav
  eval wandb: wandb/offline-run-20260627_104921-w9nn7k1z
  payload_bpp: 0.011083
  full_stream_bpp: 0.071538
  topk_hit: 0.40299
  roundtrip: true

top256:
  fit wandb: wandb/offline-run-20260627_105933-bomdes9d
  eval wandb: wandb/offline-run-20260627_105951-dyeivave
  payload_bpp: 0.010849
  full_stream_bpp: 0.071294
  topk_hit: 0.50521
  roundtrip: true

top512:
  fit wandb: wandb/offline-run-20260627_111451-9l7qjl79
  eval wandb: wandb/offline-run-20260627_112242-0guschmm
  payload_bpp: 0.010722
  full_stream_bpp: 0.071172
  topk_hit: 0.62174
  roundtrip: true
  roundtrip_failure_count: 0
```

Updated active Stage 2 semantic-token codec candidate:

```text
32768-token learned top512/escape Huffman

prior:
  outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json

token prior checkpoint:
  checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt
```

Comparison:

```text
fixed_bits payload_bpp: 0.012695
previous left-context static Huffman payload_bpp: 0.011729
active learned top512 payload_bpp: 0.010722

delta vs fixed_bits: -0.001973 bpp
delta vs left-context static Huffman: -0.001007 bpp
```

Decoder-schedule correction:

```text
Initial top512 evaluation used full teacher-forced top-k sets on the encoder
side and prefix top-k sets on the decoder side. Kodak happened to roundtrip,
but DIV2K/100 exposed one token-roundtrip failure. The eval script now encodes
with the exact same prefix-by-prefix top-k schedule that the decoder rebuilds.
```

Cross-dataset checks after the correction:

```text
DIV2K first 100:
  run: 20260627_stage2_learned_topk512_escape_huffman_32768tokens_8kprior_div2k100_eval_decodersched
  wandb/offline-run-20260627_112321-7jdhov4p
  payload_bpp: 0.010425
  full_stream_bpp: 0.070870
  topk_hit: 0.64609
  roundtrip: true
  roundtrip_failure_count: 0

CLIC professional valid:
  run: 20260627_stage2_learned_topk512_escape_huffman_32768tokens_8kprior_clicval64_eval_decodersched
  wandb/offline-run-20260627_112346-zq49c8tp
  num_images: 41
  payload_bpp: 0.009906
  full_stream_bpp: 0.070339
  topk_hit: 0.71646
  roundtrip: true
  roundtrip_failure_count: 0
```

Decision:

```text
Promote top512 learned top-k/escape as the active Stage 2 semantic-token codec
candidate. The previous left-context static Huffman remains the static
ablation/baseline, not the active path.
```

Caveat:

```text
top512 improves bpp but has a heavier top-k extraction/payload-measure cost
than top256. Keep runtime in mind before locking the final paper-facing codec;
the next backend target is RDVQ-style tensor-rANS using the same decoder-known
top-k/escape events.
```

## Compact v3 Container Recheck

The compact header was tightened again after the CRC32 + short-ID pass. Version
3 keeps the same public `CoSERBitstream(header_codec="compact",
checksum_codec="crc32")` path but writes varint dimensions/lengths and table
IDs for common codec/model strings. This changes only container overhead; the
semantic payload, decoded tokens, and reconstruction remain unchanged.

Actual compress/decompress semantic-only results:

```text
Kodak 24:
  run: 20260627_stage2_learned_topk512_escape_huffman_32768tokens_8kprior_kodak_eval_decodersched_compactv3_crc32_shortid
  wandb: wandb/offline-run-20260627_130859-k3q9zxxr
  payload_bpp: 0.010721842447916666
  full_stream_bpp: 0.015482584635416666
  previous compact-v2 CRC32 short-ID full_stream_bpp: 0.021830240885416668
  roundtrip_failure_count: 0

DIV2K first 100:
  run: 20260627_stage2_learned_topk512_escape_huffman_32768tokens_8kprior_div2k100_eval_decodersched_compactv3_crc32_shortid
  wandb: wandb/offline-run-20260627_130938-fro0s4q0
  payload_bpp: 0.0104248046875
  full_stream_bpp: 0.015185546875
  roundtrip_failure_count: 0

CLIC professional valid, 41 usable crops:
  run: 20260627_stage2_learned_topk512_escape_huffman_32768tokens_8kprior_clicproval64_eval_decodersched_compactv3_crc32_shortid
  wandb: wandb/offline-run-20260627_131919-r02hxt9a
  payload_bpp: 0.009905559260670731
  full_stream_bpp: 0.014666301448170731
  roundtrip_failure_count: 0

CLIC professional+mobile valid, first 64:
  run: 20260627_stage2_learned_topk512_escape_huffman_32768tokens_8kprior_clicval64_eval_decodersched_compactv3_crc32_shortid
  wandb: wandb/offline-run-20260627_131015-kqco3khp
  payload_bpp: 0.010049819946289062
  full_stream_bpp: 0.014810562133789062
  roundtrip_failure_count: 0
```

Decision:

```text
Promote compact v3 CRC32 as the active actual-byte container for Core MVP
crop-level experiments. Old compact v1/v2 decoding remains supported so earlier
saved streams are not invalidated.
```
