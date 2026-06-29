# Stage 5 Detail High-Frequency Context Probe

Date: 2026-06-29 JST

## Question

Can the existing decoded Stage 3 detail payload help CoD-Lite condition recovery
more if the adapter receives an explicit high-frequency detail-context branch,
without adding transmitted bits?

## Implementation

Extended `CoSERToCoDLiteConditionPyramidAdapter` with:

```text
--detail-highfreq-context-branch
```

The branch computes a decoder-side high-frequency proxy from the decoded detail
context:

```text
detail_highfreq = detail_context - avg_pool3x3(detail_context)
detail_highfreq_features = proj([detail_highfreq, abs(detail_context)])
```

The projected features are zero-initialized and added to the existing detail
feature path. This uses only already-decoded CoSER detail information, so it
does not add side information or change `actual_payload_bpp` beyond the existing
counted k16 condition-control stream.

Verification after adding the branch:

```text
.venv/bin/python -m py_compile src/coserdic/models/gencodec_backbone.py \
  scripts/train_stage4_cod_lite_adapter.py \
  scripts/sweep_stage5_lora_backbone.py \
  scripts/eval_stage4_cod_lite_adapter.py

.venv/bin/python -m pytest tests/test_gencodec_backbone.py -q
24 passed
```

## Run

Training:

```text
20260629_stage5_detailcontrol_hfcontext_lora0156_all_k16_ft500_b4ga2

checkpoint:
  checkpoints/stage4_cod_lite_adapter/20260629_stage5_detailcontrol_hfcontext_lora0156_all_k16_ft500_b4ga2.pt

wandb offline:
  wandb/offline-run-20260629_225410-gmmhlrxv
```

Training diagnostic:

```text
condition_l1_mean                  0.3878
pre_control_condition_l1_mean      0.3939
control_condition_l1_delta_mean   -0.0061
LPIPS loss mean                    0.3434
DISTS loss mean                    0.2331
stage4 PSNR mean                  21.1052
```

The training condition metrics looked stronger than the plain detail-control
fusion branch, so the checkpoint was screened on the strict decoded-bitstream
limit64 evaluation.

Limit64 evaluation:

```text
20260629_stage5_detailcontrol_hfcontext_lora0156_all_k16_ft500_limit64_eval

actual_payload_bpp  0.014634
PSNR                21.6391
MS-SSIM             0.7357
LPIPS               0.3516
DISTS               0.2577
condition_l1        0.4012
```

Limit64 comparison:

| run | bpp | PSNR | MS-SSIM | LPIPS | DISTS | condition L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| old k16 anchor | 0.014631 | 21.5865 | 0.7344 | 0.3526 | 0.2552 | 0.4025 |
| detail-control fusion | 0.014632 | 21.6464 | 0.7365 | 0.3512 | 0.2577 | 0.4011 |
| detail-control DISTS-heavy | 0.014635 | 21.6937 | 0.7385 | 0.3534 | 0.2612 | 0.4010 |
| high-frequency context | 0.014634 | 21.6391 | 0.7357 | 0.3516 | 0.2577 | 0.4012 |

## Decision

Do not promote the high-frequency detail-context branch.

It is a reasonable architectural hook and it keeps the bitstream honest, but it
does not beat the simpler detail-control fusion branch on the limit64 screen.
It also repeats the same pattern seen in the previous probes:

```text
condition L1 can improve without producing a decisive final-image gain.
decoded-detail feature routing helps slightly, but it is not enough to close
the external CoD-Lite perceptual curve gap.
```

Therefore, the next mainline move should not be another adapter-side loss or
feature-routing tweak. It should change the information available to the
diffusion backbone:

```text
1. Stage4-aware detail decoder / detail-control head trained from the same
   entropy-coded residual payload
2. semantic/detail allocation sweeps evaluated after CoD-Lite decoding
3. learned semantic-conditioned residual entropy model to buy better detail at
   the same actual payload bpp
4. only then revisit stronger condition-space losses or gates
```

This keeps the CoSER-DiC novelty centered on semantic VQ tokens + detail
residual payload + actual entropy-coded streams + pretrained diffusion prior,
instead of posthoc output manipulation.
