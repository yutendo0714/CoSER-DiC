# Training Configs

Core MVP-v0 training configs live directly in this directory:

```text
train_stage1_semantic_vq.yaml
train_stage2_semantic_entropy.yaml
train_stage3_residual.yaml
train_stage4_diffusion.yaml
train_stage4_diffusion_cod.yaml
train_stage5_joint.yaml
```

`train_stage4_diffusion.yaml` is the default lightweight Core MVP path based on
CoD-Lite pretraining. `train_stage4_diffusion_cod.yaml` is a parallel heavy
track for CoD backbone / teacher / upper-bound experiments and must not replace
the CoSER semantic/detail actual bitstream.

Post-MVP extensions are parked under `post_mvp/` and should not be run until
the Core MVP success criteria are met.
