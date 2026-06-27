# Stage 2 Learned Token Prior Probe

Date: 2026-06-27  
Status: Active Stage 2 semantic-token codec candidate after 32768-token actual-bpp sweep  
Parent policy: `docs/research/design_decisions/003_official_implementation_reference_policy.md`

## Purpose

Stage 2 static token priors proved that the semantic VQ stream is compressible
with actual transmitted bytes. The next Core MVP question is whether a simple
decoder-rebuildable learned prior can predict semantic tokens better than the
static left-context Huffman baseline without changing the CoSER-DiC novelty
axis.

The probe intentionally keeps the model simple:

```text
semantic VQ token grid
  -> raster-causal token sequence
  -> small causal Transformer prior
  -> held-out token cross-entropy / top-k hit rates
```

The first 4096-token run was an entropy-model probe only. The 32768-token
update below now emits actual bitstreams with exact token roundtrip and beats
the previous left-context static Huffman baseline on Kodak.

## Implementation

```text
src/coserdic/models/token_prior.py
scripts/export_stage1_semantic_tokens.py
scripts/train_stage2_token_prior.py
tests/test_token_prior.py
```

The current prior is a CoSER-owned raster-causal Transformer:

```text
context length: 64 tokens (8x8)
vocab size: 8192 + BOS
default d_model: 256
default layers: 4
default heads: 4
```

The implementation follows the official-reference policy: RDVQ remains the L2
reference for practical learned token entropy coding, but this model and its
training/evaluation scripts are maintained as CoSER-DiC code.

## Token Exports

Run:

```text
20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_best
wandb/offline-run-20260627_100942-jf2j05ir
```

Artifacts:

```text
outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_best/semantic_tokens.pt
outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_4096_oi_div2k_from_stage1_best/histograms.pt
```

Summary:

```text
num_images: 4096
token_shape: 8x8
total_tokens: 262144
active_codes: 7520 / 8192
global_entropy_bits: 11.9197
```

Larger export:

```text
20260627_stage2_semantic_tokens_32768_oi_div2k_from_stage1_best
wandb/offline-run-20260627_104221-imbu1ilj

outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_32768_oi_div2k_from_stage1_best/semantic_tokens.pt
outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_32768_oi_div2k_from_stage1_best/histograms.pt
```

Summary:

```text
num_images: 32768
token_shape: 8x8
total_tokens: 2097152
active_codes: 8025 / 8192
global_entropy_bits: 11.9390
```

## 4096-Token Probe Results

Best short probe:

```text
20260627_stage2_token_prior_tf256_l4_3kstep_4096tokens_probe
wandb/offline-run-20260627_101223-ox76o7pf

best val bits/token: 11.2619
final val bits/token: 11.2871
final top1 / top5 / top64: 0.0310 / 0.0777 / 0.2231
```

Longer probes exposed overfitting or late instability:

```text
20260627_stage2_token_prior_tf256_l4_10kstep_4096tokens_probe
  best val bits/token: 11.2796
  final val bits/token: 12.2071

20260627_stage2_token_prior_tf256_l4_5kstep_4096tokens_unscaleclip_probe
  best val bits/token: 11.2728
  final val bits/token: 18.9196
```

The 5k probe includes the AMP fix where gradients are unscaled before gradient
clipping. The best point remained useful, but late training still degraded.

## Initial 4096-Token Decision

Do not promote the 4096-token learned prior as the active Stage 2 codec.

Reason:

```text
The learned prior result is estimated cross-entropy, not actual transmitted
payload bpp from compress/decompress. The project selection criterion remains
real bitstream bytes with exact token roundtrip.
```

Current active actual-bpp codec:

```text
4096-calibrated left-context static Huffman
Kodak payload_bpp: 0.011729
Kodak full_stream_bpp: 0.071437
token_roundtrip: true
```

Interpretation:

```text
The learned prior is promising enough to justify building the actual learned
entropy backend. It is not reliable enough to justify long blind training on the
4096-token export alone.
```

## Actual Top-k/Escape Bridge

Implemented a CoSER-owned actual-byte bridge inspired by RDVQ's top-k/escape
entropy path:

```text
src/coserdic/entropy/topk_escape.py
scripts/fit_stage2_learned_topk_escape_prior.py
scripts/eval_stage2_learned_topk_escape_bitstream.py
tests/test_topk_escape.py
```

Coding rule:

```text
decoder rebuilds top-k candidates from causal token history
payload stores Huffman-coded top-k rank if target is in top-k
payload stores ESCAPE + fixed-width raw token id otherwise
```

This is not the final rANS backend, but it does satisfy the real bitstream
boundary: transmitted bytes are packed into `CoSERBitstream`, decoded
sequentially from decoder-rebuilt top-k sets, and checked by exact token
roundtrip.

Implementation correction:

```text
The encoder now builds top-k sets with the same prefix-by-prefix schedule used
by the decoder. A full teacher-forced Transformer pass can differ by tiny
amounts near a top-k boundary and caused one DIV2K/100 roundtrip failure for
top512. The decoder-schedule evaluation restored exact roundtrip on Kodak,
DIV2K/100, and CLIC professional valid.
```

4096-token export calibration:

```text
top64:
  run: 20260627_stage2_learned_topk64_escape_huffman_fit_4096tokens_3kprior
  wandb: wandb/offline-run-20260627_102714-m22raqxg
  topk_hit_rate: 0.41695
  payload_bpp: 0.010615

top128:
  run: 20260627_stage2_learned_topk128_escape_huffman_fit_4096tokens_3kprior
  wandb: wandb/offline-run-20260627_102828-8xrjmewu
  topk_hit_rate: 0.51890
  payload_bpp: 0.010279

top256:
  run: 20260627_stage2_learned_topk256_escape_huffman_fit_4096tokens_3kprior
  wandb: wandb/offline-run-20260627_102952-f129a8wl
  topk_hit_rate: 0.63268
  payload_bpp: 0.010031
```

Kodak actual compress/decompress evaluation:

```text
top64:
  run: 20260627_stage2_learned_topk64_escape_huffman_3kprior_kodak_eval
  wandb: wandb/offline-run-20260627_102741-sl976cl1
  payload_bpp: 0.012380
  full_stream_bpp: 0.072891
  topk_hit_rate: 0.18750
  token_roundtrip: true
  delta vs fixed_bits: -0.000315
  delta vs left-context static Huffman: +0.000651

top128:
  run: 20260627_stage2_learned_topk128_escape_huffman_3kprior_kodak_eval
  wandb: wandb/offline-run-20260627_102845-7jwvagp4
  payload_bpp: 0.012161
  full_stream_bpp: 0.072662
  topk_hit_rate: 0.24870
  token_roundtrip: true
  delta vs fixed_bits: -0.000534
  delta vs left-context static Huffman: +0.000432

top256:
  run: 20260627_stage2_learned_topk256_escape_huffman_3kprior_kodak_eval
  wandb: wandb/offline-run-20260627_103009-gyyh3v9b
  payload_bpp: 0.012421
  full_stream_bpp: 0.072927
  topk_hit_rate: 0.32617
  token_roundtrip: true
  delta vs fixed_bits: -0.000275
  delta vs left-context static Huffman: +0.000692
```

4096-token updated decision:

```text
The learned top-k/escape bridge is functional and beats fixed_bits on Kodak,
but it still underperforms the 4096-calibrated left-context static Huffman.

Best learned actual-bpp probe so far:
  top128 payload_bpp: 0.012161

Active actual-bpp baseline remains:
  left-context static Huffman payload_bpp: 0.011729
```

Diagnosis:

```text
The 4096-token training/calibration distribution is much easier than Kodak for
the learned prior. Top128 gives the best current tradeoff; top256 gains hit
rate but loses to rank-event coding overhead and weaker generalization.
```

## 32768-Token Update

The recommended data/model step was executed with a larger OpenImages + DIV2K
semantic-token export and the same simple causal Transformer. Training script
support was expanded with gradient clipping, label smoothing hooks, bounded
validation, early stopping, and richer W&B/config checkpoint metadata.

Token prior:

```text
20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es
wandb/offline-run-20260627_104246-nebvdh5a

checkpoint:
  checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt

completed_steps: 8000
stopped_early: false
best_step: 8000
best/final val bits/token: 9.8837
final top1 / top5 / top64: 0.0494 / 0.1294 / 0.3615
```

This is a major cross-entropy improvement over the 4096-token probe
(`11.2619 -> 9.8837` bits/token), and the improvement transfers to actual
payload bytes.

32768-token calibration with the actual top-k/escape Huffman bridge:

```text
top64:
  run: 20260627_stage2_learned_topk64_escape_huffman_fit_32768tokens_8kprior
  wandb: wandb/offline-run-20260627_105153-n922o2xj
  topk_hit_rate: 0.41836
  mean_payload_bpp: 0.010602

top128:
  run: 20260627_stage2_learned_topk128_escape_huffman_fit_32768tokens_8kprior
  wandb: wandb/offline-run-20260627_104902-nbwamsav
  topk_hit_rate: 0.51599
  mean_payload_bpp: 0.010287

top256:
  run: 20260627_stage2_learned_topk256_escape_huffman_fit_32768tokens_8kprior
  wandb: wandb/offline-run-20260627_105933-bomdes9d
  topk_hit_rate: 0.62270
  mean_payload_bpp: 0.010066

top512:
  run: 20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior
  wandb: wandb/offline-run-20260627_111451-9l7qjl79
  topk_hit_rate: 0.73380
  mean_payload_bpp: 0.009801
```

Kodak actual compress/decompress:

```text
top64:
  run: 20260627_stage2_learned_topk64_escape_huffman_32768tokens_8kprior_kodak_eval
  wandb: wandb/offline-run-20260627_105211-l9p5fys4
  payload_bpp: 0.011383
  full_stream_bpp: 0.071849
  topk_hit_rate: 0.31510
  token_roundtrip: true
  delta vs left-context static Huffman: -0.000346

top128:
  run: 20260627_stage2_learned_topk128_escape_huffman_32768tokens_8kprior_kodak_eval
  wandb: wandb/offline-run-20260627_104921-w9nn7k1z
  payload_bpp: 0.011083
  full_stream_bpp: 0.071538
  topk_hit_rate: 0.40299
  token_roundtrip: true
  delta vs left-context static Huffman: -0.000646

top256:
  run: 20260627_stage2_learned_topk256_escape_huffman_32768tokens_8kprior_kodak_eval
  wandb: wandb/offline-run-20260627_105951-dyeivave
  payload_bpp: 0.010849
  full_stream_bpp: 0.071294
  topk_hit_rate: 0.50521
  token_roundtrip: true
  delta vs left-context static Huffman: -0.000880

top512:
  run: 20260627_stage2_learned_topk512_escape_huffman_32768tokens_8kprior_kodak_eval
  decoder-schedule rerun: 20260627_stage2_learned_topk512_escape_huffman_32768tokens_8kprior_kodak_eval_decodersched
  wandb: wandb/offline-run-20260627_112242-0guschmm
  payload_bpp: 0.010722
  full_stream_bpp: 0.071172
  topk_hit_rate: 0.62174
  token_roundtrip: true
  roundtrip_failure_count: 0
  delta vs left-context static Huffman: -0.001007
```

Cross-dataset decoder-schedule checks:

```text
DIV2K first 100:
  run: 20260627_stage2_learned_topk512_escape_huffman_32768tokens_8kprior_div2k100_eval_decodersched
  wandb: wandb/offline-run-20260627_112321-7jdhov4p
  payload_bpp: 0.010425
  full_stream_bpp: 0.070870
  topk_hit_rate: 0.64609
  token_roundtrip: true
  roundtrip_failure_count: 0

CLIC professional valid:
  run: 20260627_stage2_learned_topk512_escape_huffman_32768tokens_8kprior_clicval64_eval_decodersched
  wandb: wandb/offline-run-20260627_112346-zq49c8tp
  num_images: 41
  payload_bpp: 0.009906
  full_stream_bpp: 0.070339
  topk_hit_rate: 0.71646
  token_roundtrip: true
  roundtrip_failure_count: 0
```

## Active Decision

Promote the 32768-token learned top512/escape Huffman bridge as the active
Stage 2 semantic-token codec candidate:

```text
prior:
  outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_8kprior/learned_topk_escape_huffman_prior.json

token prior checkpoint:
  checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf256_l4_8kstep_32768tokens_lr1e4_do02_es.pt

Kodak actual payload_bpp: 0.010722
Kodak full_stream_bpp: 0.071172
token_roundtrip: true
DIV2K/100 payload_bpp: 0.010425
CLIC professional valid payload_bpp: 0.009906
```

Reason:

```text
It is the best actual transmitted payload/full-stream result so far, preserves
exact sequential token roundtrip, and beats both fixed_bits and the previous
left-context static Huffman baseline.
```

Keep these caveats:

```text
1. top512 has a heavier top-k extraction and payload-measure pass than top256.
2. The current bridge is Huffman-coded rank/escape, not final tensor-rANS.
3. The prior is still trained only on Stage 1 semantic tokens, so detail-stream
   integration can change the optimal semantic bitrate tradeoff.
```

## Next Actions

Near-term:

```text
1. Evaluate the active top512 codec on DIV2K-val/CLIC if available, still using
   actual compress/decompress bpp.
2. Run a narrow top-k sweep around 512 only if cross-dataset results suggest
   the overhead/hit-rate optimum has not saturated.
3. Prototype an RDVQ-style tensor-rANS backend for the same decoder-rebuilt
   top-k/escape events; do not change the CoSER semantic-token interface.
4. Start Stage 3 residual coding against this top512 semantic stream, keeping
   semantic payload bpp and detail payload bpp reported separately.
```
