# Baseline Configs

Core MVP-v0 baseline policy is defined in `registry.yaml`.
Official-implementation reference boundaries are defined in
`official_reference_map.yaml` and
`docs/research/design_decisions/003_official_implementation_reference_policy.md`.

Priority A:

- CompressAI hyperprior / Cheng2020 anchors
- CoD-Lite rate checkpoints
- RDVQ
- GLC
- StableCodec

Only `CoD_Lite_pretrain.pt` may be used as a default CoSER-DiC initialization
candidate, and only for the diffusion decoder backbone. Other pretrained models
are baselines or references.

CoSER-DiC implementation must stay under `src/coserdic` as a new codec. External
official repositories may be used for baselines, sanity checks, design reference,
and narrow initialization only when the reference map explicitly allows it.
