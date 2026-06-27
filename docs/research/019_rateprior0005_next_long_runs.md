# Rateprior0005 Next Long Runs

Date: 2026-06-27

## Current State

The RDVQ-inspired Stage 1 rate-prior candidate is promising but not promoted
yet. It improves Stage 3 b4 perceptual metrics versus the ft500 topk512 branch
while slightly lowering actual payload bpp:

```text
rateprior0005 topk512 b4, 8192 residual calibration:
  actual_payload_bpp: 0.017592
  semantic/detail bpp: 0.012370 / 0.005222
  none LPIPS / DISTS:     0.614341 / 0.383102
  unsharp2 LPIPS / DISTS: 0.584765 / 0.372274
```

It still does not beat the stronger ft500 d384/l6 topk2048 entropy setting on
rate:

```text
ft500 d384/l6 topk2048 b4:
  actual_payload_bpp: 0.015850
  semantic/detail bpp: 0.010627 / 0.005223
```

The 32768-token export for the candidate is already complete:

```text
tokens:
  outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_32768_oi_div2k_from_stage1_rateprior0005_tau01_vq005_ft500_fullroots/semantic_tokens.pt

summary:
  active_codes: 8126
  global_entropy_bits: 12.268169
  ft500 baseline entropy: 12.281880
```

## 1. Train Stronger Token Prior

Run this as the next long job:

```bash
.venv/bin/python scripts/train_stage2_token_prior.py \
  --tokens outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_32768_oi_div2k_from_stage1_rateprior0005_tau01_vq005_ft500_fullroots/semantic_tokens.pt \
  --batch-size 64 \
  --num-workers 2 \
  --max-steps 12000 \
  --eval-every 500 \
  --lr 1e-4 \
  --weight-decay 0.01 \
  --d-model 384 \
  --num-layers 6 \
  --num-heads 6 \
  --dropout 0.2 \
  --grad-clip-norm 1.0 \
  --early-stop-patience 5 \
  --early-stop-min-delta 0.001 \
  --wandb-mode offline \
  --run-name 20260627_stage2_token_prior_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_tau01_vq005_ft500_lr1e4_do02_es
```

Expected checkpoint:

```text
checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_tau01_vq005_ft500_lr1e4_do02_es.pt
```

## 2. Fit Learned Semantic Priors

Run both topk512 and topk2048. The short d256/l4 prior preferred topk512 on
actual payload, but the stronger d384/l6 prior may change that tradeoff.

```bash
.venv/bin/python scripts/fit_stage2_learned_topk_escape_prior.py \
  --tokens outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_32768_oi_div2k_from_stage1_rateprior0005_tau01_vq005_ft500_fullroots/semantic_tokens.pt \
  --token-prior-checkpoint checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_tau01_vq005_ft500_lr1e4_do02_es.pt \
  --topk 512 \
  --batch-size 64 \
  --num-workers 2 \
  --wandb-mode offline \
  --run-name 20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_stage1_rateprior0005_tau01_vq005_ft500_tf384_l6_12k
```

```bash
.venv/bin/python scripts/fit_stage2_learned_topk_escape_prior.py \
  --tokens outputs/stage2_semantic_tokens/20260627_stage2_semantic_tokens_32768_oi_div2k_from_stage1_rateprior0005_tau01_vq005_ft500_fullroots/semantic_tokens.pt \
  --token-prior-checkpoint checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_tau01_vq005_ft500_lr1e4_do02_es.pt \
  --topk 2048 \
  --batch-size 64 \
  --num-workers 2 \
  --wandb-mode offline \
  --run-name 20260627_stage2_learned_topk2048_escape_huffman_fit_32768tokens_stage1_rateprior0005_tau01_vq005_ft500_tf384_l6_12k
```

## 3. Evaluate Stage 3

Use the 8192-calibrated candidate residual prior:

```text
outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_stage1_rateprior0005_tau01_vq005_ft500/static_residual_grid_semantic_position_leftctx_huffman_prior.json
```

For each fitted semantic prior, run both no postprocess and unsharp2. Example
for topk512:

```bash
.venv/bin/python scripts/eval_stage3_uniform_residual_bitstream.py \
  --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500.pt \
  --semantic-prior outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_stage1_rateprior0005_tau01_vq005_ft500_tf384_l6_12k/learned_topk_escape_huffman_prior.json \
  --token-prior-checkpoint checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_tau01_vq005_ft500_lr1e4_do02_es.pt \
  --detail-prior outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_stage1_rateprior0005_tau01_vq005_ft500/static_residual_grid_semantic_position_leftctx_huffman_prior.json \
  --eval-protocol coser_common_lic \
  --batch-size 4 \
  --num-workers 4 \
  --detail-codec semantic_position_leftctx_huffman \
  --detail-downsample-factor 32 \
  --detail-bits 4 \
  --detail-range 0.25 \
  --stream-header-codec compact \
  --stream-checksum-codec crc32 \
  --compute-perceptual \
  --wandb-mode offline \
  --run-name 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_rateprior0005_tau01_vq005_ft500_tf384_topk512_32kprior_8192resid_perceptual
```

```bash
.venv/bin/python scripts/eval_stage3_uniform_residual_bitstream.py \
  --checkpoint checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500.pt \
  --semantic-prior outputs/stage2_learned_entropy/20260627_stage2_learned_topk512_escape_huffman_fit_32768tokens_stage1_rateprior0005_tau01_vq005_ft500_tf384_l6_12k/learned_topk_escape_huffman_prior.json \
  --token-prior-checkpoint checkpoints/stage2_token_prior/20260627_stage2_token_prior_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_tau01_vq005_ft500_lr1e4_do02_es.pt \
  --detail-prior outputs/stage3_residual_entropy/20260627_stage3_residual_semposleft_g4_sm0_d32_b4_r025_8192calib_stage1_rateprior0005_tau01_vq005_ft500/static_residual_grid_semantic_position_leftctx_huffman_prior.json \
  --eval-protocol coser_common_lic \
  --batch-size 4 \
  --num-workers 4 \
  --detail-codec semantic_position_leftctx_huffman \
  --detail-downsample-factor 32 \
  --detail-bits 4 \
  --detail-range 0.25 \
  --stream-header-codec compact \
  --stream-checksum-codec crc32 \
  --compute-perceptual \
  --decoder-postprocess unsharp3x3 \
  --decoder-postprocess-strength 2.0 \
  --wandb-mode offline \
  --run-name 20260627_stage3_b4_semposleft_g4_coser_common_lic_stage1_rateprior0005_tau01_vq005_ft500_tf384_topk512_32kprior_8192resid_pp_unsharp200_perceptual
```

## Promotion Rule

Promote only if the d384/l6 candidate materially closes the semantic bpp gap
while retaining the perceptual gain:

```text
target:
  actual_payload_bpp near or below 0.015850
  LPIPS/DISTS below the ft500 d384/l6 topk2048 branch
  semantic and detail roundtrip both true
```
