# Stage 1 Semantic VQ Progress

Date: 2026-06-27 JST

Core MVP reference:

```text
docs/research/design_decisions/000_external_repos_and_pretrained_policy_mvp.md
```

## Why Stage 1 First

Core MVP-v0 needs to establish that a scratch-trained semantic VQ stream can
carry global structure before detail residuals or diffusion are added. This
keeps CoSER-DiC from becoming "RDVQ + CoD-Lite" and preserves the main novelty:

```text
semantic structure / detail-critical residual / stochastic texture are separated
under entropy constraints.
```

## Implemented

- `SemanticEncoder`
- `DifferentiableVQ`
- `SemanticAuxiliaryDecoder`
- `SemanticVQAutoEncoder`
- Stage 1 loss:
  - L1
  - MS-SSIM
  - optional LPIPS
  - VQ commitment
  - weak code usage pressure
- Stage 1 training script:
  - W&B logging
  - checkpoint saving
  - visual sample saving
  - experiment card generation
  - non-finite loss guard

## Smoke Runs

Failed run:

```text
20260627_stage1_semantic_vq_debug_smoke
```

Failure:

```text
loss_total = NaN
perplexity = NaN
```

Cause:

```text
VQ perplexity was computed under CUDA autocast / fp16. The 1e-10 epsilon
underflowed, causing 0 * log(0) style NaNs.
```

Fix:

```text
Compute VQ distances, one-hot assignment, EMA update, and perplexity in fp32.
Return quantized tensor in the original activation dtype.
```

Successful run:

```text
20260627_stage1_semantic_vq_debug_smoke_fp32vq
```

Command:

```bash
source .venv/bin/activate
python scripts/train_stage1_semantic_vq.py \
  --debug-small \
  --disable-lpips \
  --max-steps 5 \
  --batch-size 2 \
  --num-workers 0 \
  --log-every 1 \
  --sample-every 5 \
  --wandb-mode offline \
  --run-name 20260627_stage1_semantic_vq_debug_smoke_fp32vq
```

Summary:

```text
loss_total: 0.44081076979637146
loss_l1_sem: 0.26023316383361816
loss_ms_ssim_sem: 0.8116093277931213
loss_vq: 0.017163332551717758
perplexity: 9.154203414916992
dead_code_ratio: 0.8828125
used_codes: 15
psnr_sem: 10.283122062683105
step: 5
```

Artifacts:

```text
checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_debug_smoke_fp32vq.pt
outputs/stage1_semantic_vq/20260627_stage1_semantic_vq_debug_smoke_fp32vq/
docs/experiments/20260627_stage1_semantic_vq_debug_smoke_fp32vq.md
wandb/offline-run-20260627_023844-qkpz3cau
```

The metrics are not meaningful as research results because this is a 5-step
debug-small smoke run. They only verify that the Stage 1 training path is alive.

## 03:10 JST Update

The direct hard-VQ path was unstable. Important negative results are preserved
under `docs/experiments/failures/`.

Observed failures:

- EMA VQ collapsed to 2-3 active codes before semantic structure formed.
- Non-EMA hard VQ without AE pretraining looked acceptable at 100 steps, but a
  1k run collapsed into near-constant output on validation.
- Quantize/VQ warmup with tiny nonzero VQ pressure still collapsed.
- soft-ST and normalized cosine VQ collapsed faster in 1k probes.
- Normalized continuous latents were bad for the auxiliary decoder.

Critical diagnosis:

```text
Stage 1 needs a real continuous-AE warm start. Even tiny early VQ pressure can
push the encoder/decoder into saturated constant-color solutions.
```

## Current Implemented Research Path

Training script additions:

- `--fixed-quantize-mix`
- loss-weight CLI overrides
- `--grad-clip-norm`
- `--init-checkpoint`
- `--no-init-load-codebook`
- `--reinit-codebook-from-encoder`
- encoder/decoder/output feature statistics in W&B

Analysis script addition:

- `--quantize-mix`, so continuous and hard-VQ checkpoints can be evaluated
  separately.

Default Stage 1 config now uses the conservative baseline:

```text
non-EMA / hard ST / native unnormalized latent / no implicit warmup
```

soft-ST, normalized VQ, and EMA remain explicit ablations, not defaults.
MS-SSIM is also disabled as a training loss for now because it produced
non-finite Stage 1 gradients at initialization; it remains an evaluation metric.

## Best Current Stage 1 Route

Step A: pure continuous AE warm start.

Run:

```text
20260627_stage1_semantic_vq_fullshape_1k_pure_ae_l1_probe
```

Command key:

```bash
python scripts/train_stage1_semantic_vq.py \
  --fixed-quantize-mix 0 \
  --loss-ms-ssim-sem 0 \
  --loss-vq 0 \
  --loss-codebook-usage 0 \
  --grad-clip-norm 1.0 \
  --no-soft-st \
  --no-normalize-latent
```

Train summary:

```text
PSNR: 18.40 dB
L1: 0.0847
h_std: 2.05
```

Kodak continuous analysis:

```text
20260627_stage1_semantic_vq_1k_pure_ae_l1_kodak_continuous_analysis

mean_psnr_sem: 19.77 dB
mean_l1_sem: 0.0769
mean_ms_ssim_sem: 0.6263
latent_std: 1.69
quantized_std before codebook init: 0.49
```

Step B: initialize the VQ codebook from AE encoder features and switch to hard
VQ.

Run:

```text
20260627_stage1_semantic_vq_fullshape_1k_hardvq_from_ae_codeinit_l1
```

Command key:

```bash
python scripts/train_stage1_semantic_vq.py \
  --init-checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_1k_pure_ae_l1_probe.pt \
  --no-init-load-codebook \
  --reinit-codebook-from-encoder \
  --codebook-init-max-vectors 65536 \
  --fixed-quantize-mix 1 \
  --loss-ms-ssim-sem 0 \
  --loss-vq 0.25 \
  --loss-codebook-usage 0.01 \
  --grad-clip-norm 1.0 \
  --lr 5e-5 \
  --no-soft-st \
  --no-normalize-latent
```

Train summary:

```text
PSNR: 19.29 dB
L1: 0.0669
perplexity: 141.06
used_codes per batch: 183
VQ MSE: 0.167
```

Kodak hard-VQ analysis:

```text
20260627_stage1_semantic_vq_1k_hardvq_codeinit_l1_kodak_analysis

mean_psnr_sem: 19.66 dB
mean_l1_sem: 0.0769
mean_ms_ssim_sem: 0.6166
active_codes_global: 766 / 8192
global_code_perplexity: 551.55
mean_per_image_used_codes: 52.33 / 64 tokens
mean_vq_mse: 0.101
latent_std: 1.16
quantized_std: 1.15
```

## Current Interpretation

Codebook initialization from encoder features is now part of the Core MVP Stage
1 recipe. It preserves the novelty axis because the semantic VQ is still
scratch-trained; no external tokenizer or pretrained compressor is used.

The current semantic stream is a low-frequency structure carrier, not a final
reconstruction model. This is aligned with Core MVP because detail fidelity is
reserved for the detail residual stream and stochastic texture decoder.

## 03:25 JST Update: 5k Warm Start Completed

The recommended medium runs were executed and produced a stronger Stage 1
baseline.

Step A: 5k pure continuous AE warm start.

Run:

```text
20260627_stage1_semantic_vq_fullshape_5k_pure_ae_l1
```

Kodak continuous analysis:

```text
20260627_stage1_semantic_vq_5k_pure_ae_l1_kodak_continuous_analysis

mean_psnr_sem: 21.33 dB
mean_l1_sem: 0.0624
mean_ms_ssim_sem: 0.7650
active_codes_global by nearest assignment: 361 / 8192
global_code_perplexity by nearest assignment: 75.28
latent_std: 1.88
quantized_std before codebook init: 0.49
```

Interpretation:

```text
The 5k AE improves Kodak PSNR by roughly +1.57 dB over the 1k AE warm start.
It is now the preferred initialization checkpoint for Stage 1 semantic VQ.
```

Step B: encoder-sampled codebook initialization followed by 5k hard VQ.

Run:

```text
20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_l1
```

Kodak hard-VQ analysis:

```text
20260627_stage1_semantic_vq_5k_hardvq_from_5k_ae_codeinit_l1_kodak_analysis

mean_psnr_sem: 20.12 dB
mean_l1_sem: 0.0725
mean_ms_ssim_sem: 0.6373
active_codes_global: 729 / 8192
global_code_entropy_bits: 9.04 bits/token
global_code_perplexity: 527.09
mean_per_image_used_codes: 53.08 / 64 tokens
mean_vq_mse: 0.216
latent_std: 1.25
quantized_std: 1.20
```

Comparison to 1k hard-VQ:

```text
Kodak PSNR: 19.66 -> 20.12 dB
Kodak L1: 0.0769 -> 0.0725
Kodak MS-SSIM: 0.6166 -> 0.6373
global code perplexity: 551.55 -> 527.09
active codes: 766 -> 729
```

Interpretation:

```text
The 5k hard-VQ checkpoint is the current Stage 1 baseline. It improves
reconstruction quality without increasing token usage. The continuous-to-hard
gap is still significant: 21.33 -> 20.12 dB and 0.765 -> 0.637 MS-SSIM.
Closing this gap is the main Stage 1 improvement target.
```

## Actual Bitstream Check

Implemented:

```text
src/coserdic/entropy/semantic_tokens.py
scripts/eval_stage1_semantic_bitstream.py
tests/test_semantic_tokens.py
```

Purpose:

```text
Measure actual transmitted bytes through CoSERBitstream, not estimated entropy.
This is a deterministic Stage 1 baseline for semantic tokens; learned entropy
modeling is still Stage 2.
```

Run:

```text
20260627_stage1_semantic_vq_5k_hardvq_kodak_actual_bitstream_eval
```

Kodak 256x256 crop results:

```text
tokens/image: 64
codebook: 8192

fixed_bits payload:
  token payload bytes/image: 104
  token payload bpp: 0.01270
  full CoSERBitstream bytes/image: 609
  full actual transmitted bpp: 0.07434

raw_uint16 payload:
  token payload bytes/image: 128
  token payload bpp: 0.01563
  full actual transmitted bpp: 0.07764

zlib is not useful at this tiny per-image payload size:
  zlib_fixed_bits payload bpp: 0.01404
  zlib_uint16 payload bpp: 0.01660
```

Important distinction:

```text
The full actual bpp includes the current JSON/checksum CoSERBitstream container
header. This is the correct transmitted-byte number for MVP audits. The payload
bpp is the target that Stage 2 learned entropy coding should beat.
```

## Current Stage 1 Baseline

Use:

```text
checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_l1.pt
```

Recipe:

```text
1. Train pure continuous AE with quantize_mix=0, L1 only, no VQ pressure.
2. Reinitialize codebook from encoder latents sampled over the train loader.
3. Fine-tune hard VQ with quantize_mix=1, non-EMA, unnormalized latent,
   VQ weight 0.25, code usage weight 0.02, lr 5e-5.
4. Evaluate both reconstruction and actual bitstream bpp.
```

Near-term research targets:

```text
1. Reduce continuous-to-hard gap without code collapse.
2. Add learned semantic token entropy model and compare against fixed_bits
   payload bpp = 0.01270 on Kodak 256 crops.
3. Start detail residual latent stream only after semantic token actual-bpp
   evaluation is stable.
```

## 3k Low-VQ Hard-VQ Probe

Motivation:

```text
The 5k hard-VQ baseline used VQ weight 0.25 and code usage weight 0.02.
It improved over the 1k model, but still left a large continuous-to-hard gap.
This probe reduces quantization pressure while keeping hard VQ from step 0
after AE codebook initialization:

VQ weight: 0.25 -> 0.10
code usage weight: 0.02 -> 0.01
lr: 5e-5
quantize_mix: 1
```

Run:

```text
20260627_stage1_semantic_vq_fullshape_3k_hardvq_from_5k_ae_codeinit_vq010_usage001
```

Checkpoint:

```text
checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_3k_hardvq_from_5k_ae_codeinit_vq010_usage001.pt
```

Kodak hard-VQ analysis:

```text
20260627_stage1_semantic_vq_3k_hardvq_from_5k_ae_codeinit_vq010_usage001_kodak_analysis

mean_psnr_sem: 20.37 dB
mean_l1_sem: 0.0696
mean_ms_ssim_sem: 0.6634
active_codes_global: 1022 / 8192
global_code_entropy_bits: 9.74 bits/token
global_code_perplexity: 856.32
mean_per_image_used_codes: 56.13 / 64 tokens
mean_vq_mse: 0.532
latent_std: 1.85
quantized_std: 1.75
```

Comparison to the 5k hard-VQ baseline:

```text
Kodak PSNR: 20.12 -> 20.37 dB
Kodak L1: 0.0725 -> 0.0696
Kodak MS-SSIM: 0.6373 -> 0.6634
active codes: 729 -> 1022
global code entropy: 9.04 -> 9.74 bits/token
global code perplexity: 527.09 -> 856.32
mean per-image used codes: 53.08 -> 56.13
mean VQ MSE: 0.216 -> 0.532
```

Actual bitstream evaluation:

```text
20260627_stage1_semantic_vq_3k_hardvq_vq010_usage001_kodak_actual_bitstream_eval

tokens/image: 64
codebook: 8192

fixed_bits payload:
  token payload bytes/image: 104
  token payload bpp: 0.01270
  full CoSERBitstream bytes/image: 609
  full actual transmitted bpp: 0.07434
  token roundtrip: true

raw_uint16 payload:
  token payload bytes/image: 128
  token payload bpp: 0.01563
  full actual transmitted bpp: 0.07764
  token roundtrip: true

zlib remains worse than fixed_bits at this payload size:
  zlib_fixed_bits payload bpp: 0.01404
  zlib_uint16 payload bpp: 0.01668
```

Interpretation:

```text
This is the new Stage 1 baseline for the Core MVP semantic token stream.
It improves reconstruction quality at the same deterministic fixed_bits
payload bpp because token count and codebook size are unchanged.

The higher token entropy/perplexity is acceptable for Stage 1 quality, but it
raises the bar for Stage 2 learned semantic entropy coding. The fixed_bits
payload remains 13 bits/token; learned token entropy must be evaluated by
actual encode/decode bytes, not estimated cross entropy alone.

The elevated VQ MSE suggests that a better next probe is not simply more VQ
pressure. More promising directions are longer low-pressure hard-VQ training,
gentle VQ-weight ramp after codebook initialization, or residual/detail stream
training that absorbs quantization error without disturbing the semantic-token
novelty axis.
```

## Current Stage 1 Baseline Update

Use:

```text
checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_3k_hardvq_from_5k_ae_codeinit_vq010_usage001.pt
```

Recipe:

```text
1. Train pure continuous AE with quantize_mix=0, L1 only, no VQ pressure.
2. Reinitialize codebook from encoder latents sampled over the train loader.
3. Fine-tune hard VQ with quantize_mix=1, non-EMA, unnormalized latent,
   VQ weight 0.10, code usage weight 0.01, lr 5e-5.
4. Evaluate reconstruction and actual transmitted CoSERBitstream bpp.
```

## Stage 1 Reference-Audit Re-evaluation

Analysis run:

```text
20260627_stage1_semantic_vq_5k_hardvq_vq010_usage001_l1only_kodak_reference_audit_analysis

wandb: wandb/offline-run-20260627_091948-9wuu12ll
doc: docs/experiments/20260627_stage1_semantic_vq_5k_hardvq_vq010_usage001_l1only_kodak_reference_audit_analysis.md
```

Key results:

```text
num_images: 24
semantic_tokens_per_image: 64
PSNR: 20.7362 dB
L1: 0.06508
MS-SSIM: 0.67577
active_codes_global: 1064 / 8192
global_code_entropy_bits: 9.8306
global_code_perplexity: 910.55
mean_per_image_used_codes: 57.17 / 64
mean_assignment_sample_entropy_bits: 0.2015
mean_assignment_avg_entropy_bits_batch: 7.6758
mean_soft_usage_entropy_bits: 7.8285
mean_commitment_mse: 0.5267
mean_codebook_mse: 0.5267
```

Bitstream run:

```text
20260627_stage1_semantic_vq_5k_hardvq_vq010_usage001_l1only_kodak_reference_audit_bitstream

wandb: wandb/offline-run-20260627_092002-kwfua59f
doc: docs/experiments/20260627_stage1_semantic_vq_5k_hardvq_vq010_usage001_l1only_kodak_reference_audit_bitstream.md
```

Actual transmitted-byte results:

```text
fixed_bits:
  token_payload_bpp: 0.012695
  full_stream_bpp: 0.074341
  token_payload_bytes/image: 104
  stream_bytes/image: 609
  token_roundtrip: true

zlib_fixed_bits:
  token_payload_bpp: 0.014023
  full_stream_bpp: 0.076279
  token_payload_bytes/image: 114.875
  stream_bytes/image: 624.875
  token_roundtrip: true
```

Interpretation:

```text
The current Stage 1 checkpoint still passes the semantic-token freeze gate:
quality is unchanged, token roundtrip is exact, code usage has not collapsed,
and actual bpp is reported from emitted bytes.

The fixed_bits payload is better than zlib_fixed_bits on this short 64-token
stream. Stage 2 should therefore target learned/token-prior entropy coding
rather than generic byte compression. The large gap between global token
entropy (~9.83 bits/token) and fixed 13 bits/token is the immediate payload
savings target.
```

## Stage 2 Static Huffman Bootstrap

Design note:

```text
docs/research/design_decisions/005_stage2_static_huffman_token_prior.md
```

Implementation:

```text
src/coserdic/entropy/static_huffman.py
scripts/fit_stage2_static_huffman_prior.py
scripts/eval_stage2_static_huffman_bitstream.py
tests/test_static_huffman.py
```

Calibration:

```text
20260627_stage2_static_huffman_prior_512calib_from_stage1_best
wandb: wandb/offline-run-20260627_092913-cb1lzc77

num_images: 256
active_codes: 4725 / 8192
global_entropy_bits: 11.6919
fixed_bits_payload_bpp: 0.012695
mean_huffman_payload_bpp: 0.011768
payload_bpp_delta_vs_fixed: -0.000927
```

Kodak evaluation:

```text
20260627_stage2_static_huffman_512calib_kodak_eval
wandb: wandb/offline-run-20260627_092928-568umx2m

static_huffman_payload_bpp: 0.012054
fixed_bits_payload_bpp: 0.012695
payload_bpp_delta_vs_fixed: -0.000641
static_huffman_full_stream_bpp: 0.074478
fixed_bits_full_stream_bpp: 0.074707
token_roundtrip: true
PSNR: 20.7363 dB
MS-SSIM: 0.67577
```

Interpretation:

```text
This is a positive Stage 2 bootstrap result. A decoder-known static categorical
token code already improves actual semantic-token payload bpp by about 5% on
Kodak without changing reconstruction quality.

The next Stage 2 target is a CoSER-owned learned/contextual semantic token
prior and then an RDVQ-style top-k/escape arithmetic or rANS coder. The JSON
audit container overhead still dominates full-stream bpp, so payload bpp and
full-stream bpp must continue to be reported separately.
```

## Stage 1 Reimplementation Decision

Decision:

```text
Do not fully rewrite Stage 1 or restart Stage 1 training from scratch right now.
The current Stage 1 remains the active CoSER-owned semantic-token baseline.
```

Reason:

```text
The current implementation already adopts the official-reference VQ essentials:
distance assignment, straight-through quantization, codebook/commitment loss
decomposition, code usage diagnostics, and actual token bitstream auditing.

Official-reference-inspired alternatives have already produced negative or
unstable results in probes: EMA from scratch, direct hard VQ without AE warm
start, soft-ST, normalized/cosine VQ, and perceptual/SSIM losses during Stage 1
training.

The Stage 2 static Huffman result further shows that the current token stream is
usable and compressible. So the highest-return path is Stage 2 learned/contextual
semantic entropy coding, not a full Stage 1 rewrite.
```

Follow-up ablations to keep open:

```text
EMA from a healthy checkpoint
RDVQ-style entropy pressure after a learned token prior exists
10k-20k low-pressure Stage 1 extension from the current best checkpoint
compact production bitstream container
```

## Official Implementation Reference Policy

Decision:

```text
CoSER-DiC should not become a direct fork or assembly of official baselines.
It remains a new codec under src/coserdic. Official implementations are used
as baselines, sanity checks, design references, and tightly scoped
initialization sources only.
```

Added policy files:

```text
docs/research/design_decisions/003_official_implementation_reference_policy.md
docs/research/design_decisions/004_stage1_semantic_vq_reference_audit.md
configs/baselines/official_reference_map.yaml
```

Stage 1 re-audit and downstream implication:

```text
Official implementation reference starts at Stage 1, not Stage 2. The current
semantic VQ route remains CoSER-owned, but it must be rechecked against RDVQ,
GLC, and CoD-Lite before being treated as fixed.

Stage 1 adopts VQ loss decomposition, entropy/codebook diagnostics, and actual
token bitstream auditing. It deliberately does not adopt RDVQ checkpoint
initialization, RDVQ tokenizer replacement, CoD-Lite condition codec as the
primary tokenizer, or GLC as a replacement for semantic/detail separation.

Later stages must preserve the Core MVP stream separation:

semantic VQ tokens -> semantic entropy
detail residual latents -> semantic-conditioned residual entropy
auxiliary reconstruction -> one/few-step diffusion refinement
actual compress/decompress bytes -> official bpp
```

Stage 1 reference-audit implementation update:

```text
Added RDVQ-style diagnostics to Stage 1:
  loss_vq_commitment
  loss_vq_codebook
  assignment_sample_entropy_bits
  assignment_avg_entropy_bits
  soft_usage_entropy_bits

Updated Stage 1 bitstream eval to record semantic token count from actual
encoder indices instead of assuming crop_size // 32.
```

Smoke verification:

```text
20260627_stage1_reference_audit_metrics_smoke

max_steps: 2
status: passed
wandb: wandb/offline-run-20260627_091543-gh2qvnt3
checkpoint: checkpoints/stage1_semantic_vq/20260627_stage1_reference_audit_metrics_smoke.pt
experiment doc: docs/experiments/20260627_stage1_reference_audit_metrics_smoke.md
```

## 5k Low-VQ Hard-VQ Extension

Motivation:

```text
The 3k low-VQ probe became the new baseline. I then tested whether the same
simple recipe benefits from longer hard-VQ fine-tuning or whether it was mainly
an early-stopping sweet spot.
```

Run:

```text
20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only
```

Checkpoint:

```text
checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
```

Recipe:

```text
1. Start from the 5k pure continuous AE checkpoint.
2. Reinitialize the codebook from encoder latents.
3. Hard-VQ fine-tune for 5k steps with:
   quantize_mix=1, VQ weight=0.10, usage weight=0.01, lr=5e-5.
4. Keep MS-SSIM and LPIPS training losses disabled.
```

Note on LPIPS:

```text
An attempted 5k run inherited lpips_sem=0.1 from the YAML config and failed at
step 1 with a non-finite gradient. The Stage 1 config now defaults LPIPS to 0.0.
LPIPS remains an explicit ablation option, but not part of the stable Core MVP
semantic-token baseline.
```

Kodak hard-VQ analysis:

```text
20260627_stage1_semantic_vq_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only_kodak_analysis

mean_psnr_sem: 20.74 dB
mean_l1_sem: 0.0651
mean_ms_ssim_sem: 0.6758
active_codes_global: 1064 / 8192
global_code_entropy_bits: 9.83 bits/token
global_code_perplexity: 910.55
mean_per_image_used_codes: 57.17 / 64 tokens
mean_vq_mse: 0.527
latent_std: 1.91
quantized_std: 1.80
```

Comparison to the 3k low-VQ baseline:

```text
Kodak PSNR: 20.37 -> 20.74 dB
Kodak L1: 0.0696 -> 0.0651
Kodak MS-SSIM: 0.6634 -> 0.6758
active codes: 1022 -> 1064
global code entropy: 9.74 -> 9.83 bits/token
global code perplexity: 856.32 -> 910.55
mean per-image used codes: 56.13 -> 57.17
mean VQ MSE: 0.532 -> 0.527
```

Actual bitstream evaluation:

```text
20260627_stage1_semantic_vq_5k_hardvq_vq010_usage001_l1only_kodak_actual_bitstream_eval

fixed_bits payload:
  token payload bytes/image: 104
  token payload bpp: 0.01270
  full CoSERBitstream bytes/image: 609
  full actual transmitted bpp: 0.07434
  token roundtrip: true

quality through actual encode/decode:
  PSNR: 20.7363 dB
  L1: 0.06508
  MS-SSIM: 0.67577
```

Interpretation:

```text
The 5k low-VQ extension is now the best Stage 1 semantic-token checkpoint.
It improves quality over both the 5k high-VQ baseline and the 3k low-VQ probe
without changing deterministic fixed_bits payload bpp.

The result strengthens the current Stage 1 recipe:
continuous AE warm start -> encoder-latent codebook init -> low-pressure
hard-VQ fine-tuning. The next Stage 1 probe should test whether a longer
10k-20k low-pressure run keeps improving or saturates, but that is a longer
training job and should be scheduled explicitly.
```

## Current Stage 1 Baseline Update 2

Use:

```text
checkpoints/stage1_semantic_vq/20260627_stage1_semantic_vq_fullshape_5k_hardvq_from_5k_ae_codeinit_vq010_usage001_l1only.pt
```

Recipe:

```text
1. Train pure continuous AE with quantize_mix=0, L1 only, no VQ pressure.
2. Reinitialize codebook from encoder latents sampled over the train loader.
3. Fine-tune hard VQ with quantize_mix=1, non-EMA, unnormalized latent,
   VQ weight 0.10, code usage weight 0.01, lr 5e-5.
4. Evaluate reconstruction and actual transmitted CoSERBitstream bpp.
```
