# Baseline Configs

Core MVP-v0 baseline policy is defined in `registry.yaml`.

Priority A:

- CompressAI hyperprior / Cheng2020 anchors
- CoD-Lite rate checkpoints
- RDVQ
- GLC
- StableCodec

Only `CoD_Lite_pretrain.pt` may be used as a default CoSER-DiC initialization
candidate, and only for the diffusion decoder backbone. Other pretrained models
are baselines or references.

