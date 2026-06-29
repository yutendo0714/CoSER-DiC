# Current Core MVP State

Date: 2026-06-28 JST  
Status: Active navigation memo for Stage 4 / Stage 5 work

## 2026-06-30 Zero-Centered Detail Quantizer Update

Added a zero-centered residual-grid quantizer for Stage 3 detail payloads. The
old 2-bit uniform residual path could not represent zero residual with an exact
zero code, which was a bad match for sparse/near-zero residual grids and made
low-bit detail-allocation probes look worse than they should.

```text
d16 b2 zero-centered Stage3, limit64:
  bpp 0.021151
  semantic/detail bpp 0.008796 / 0.012354
  PSNR 22.2556 / MS-SSIM 0.7588
  LPIPS 0.5496 / DISTS 0.3464
  detail entropy 0.2697 bits

d16 b2 uniform Stage3, limit64:
  bpp 0.024131
  PSNR 20.8772 / MS-SSIM 0.7002
  LPIPS 0.6952 / DISTS 0.3708
  detail entropy 1.0038 bits

d16 b3 uniform Stage3, limit64:
  bpp 0.025829
  PSNR 22.4673 / MS-SSIM 0.7642
  LPIPS 0.5945 / DISTS 0.3508
  detail entropy 1.3219 bits
```

Stage 4 / Stage 5 limit64 screen:

```text
d32 detail-control fusion:
  bpp 0.014632
  PSNR 21.6464 / MS-SSIM 0.7365
  LPIPS 0.3512 / DISTS 0.2577
  condition_l1 0.4011

d16 b2 zero-centered + dedicated FT400 adapter:
  bpp 0.022120
  PSNR 21.5267 / MS-SSIM 0.7361
  LPIPS 0.3523 / DISTS 0.2543
  condition_l1 0.4064

d16 b3 dedicated FT400 adapter:
  bpp 0.026810
  PSNR 21.7973 / MS-SSIM 0.7439
  LPIPS 0.3499 / DISTS 0.2499
  condition_l1 0.4090
```

Decision:

```text
Keep zero-centered residual quantization as mainline codec infrastructure.

Do not promote the d16 b2 zero-centered checkpoint as a Stage 5 result.
It fixes a real detail-payload design problem and makes low-bit residual coding
much stronger, but the Stage 4 result spends more actual payload bpp than the
d32 anchor and does not yet dominate the curve.

Next mainline work should extend this idea to d32 / dead-zone quantization,
semantic-conditioned residual entropy modeling, and Stage4-aware detail heads,
not to more posthoc RGB or adapter-only tweaks.
```

See:

```text
docs/research/design_decisions/073_stage5_zero_centered_detail_quantizer_probe_20260630.md
```

## 2026-06-29 Detail High-Frequency Context Update

Added and screened an explicit high-frequency detail-context branch for the
Stage 5 CoD-Lite pyramid adapter. It uses only decoded Stage 3 detail context
and therefore does not add bits.

```text
high-frequency detail context, limit64:
  bpp 0.014634
  PSNR 21.6391 / MS-SSIM 0.7357
  LPIPS 0.3516 / DISTS 0.2577
  condition_l1 0.4012

plain detail-control fusion, limit64:
  bpp 0.014632
  PSNR 21.6464 / MS-SSIM 0.7365
  LPIPS 0.3512 / DISTS 0.2577
  condition_l1 0.4011
```

Decision:

```text
Do not promote the high-frequency detail-context branch.

It improves the training condition target, but does not beat the simpler
detail-control fusion branch on final-image limit64 metrics. This reinforces
the current diagnosis: adapter-side routing and loss tweaks are not enough.

Next mainline work should change the information available to the diffusion
backbone through Stage4-aware detail representation, semantic/detail allocation,
or learned detail entropy modeling.
```

See:

```text
docs/research/design_decisions/072_stage5_detail_highfreq_context_probe_20260629.md
```

## 2026-06-29 Detail-Control Fusion Update

Added a condition-aware detail-control branch to the Stage 5 CoD-Lite pyramid
adapter. It uses the existing decoded detail payload, so it does not add bits.

```text
balanced detail-control fusion:
  full552 bpp 0.014977
  PSNR 21.2634 / MS-SSIM 0.7137
  LPIPS 0.3649 / DISTS 0.2597
  condition_l1 0.4014

DISTS-heavy detail-control fusion:
  full552 bpp 0.014978
  PSNR 21.3102 / MS-SSIM 0.7158
  LPIPS 0.3682 / DISTS 0.2630
  condition_l1 0.4014
```

Decision:

```text
Keep the balanced detail-control fusion implementation as useful mainline
infrastructure, but do not promote either run as a Stage 5 result.

The new branch gives small fidelity / LPIPS improvements at the same actual
payload bpp, but DISTS remains worse than the old k16 anchor. DISTS-heavy
loss reweighting improves PSNR/MS-SSIM further while worsening LPIPS/DISTS.

Next work should change the transmitted/decoded detail representation and
Stage4-aware detail allocation, not keep tuning loss weights around the same
detail payload.
```

See:

```text
docs/research/design_decisions/071_stage5_detailcontrol_fusion_probe_20260629.md
```

## 2026-06-29 Fixed-Target / Base-Fixed / Selector Update

New Stage 5 probes after the control-aware FT400 result:

```text
base-fixed condition control:
  implemented condition_base_affine_basis for train/eval
  limit64 complement FT500:
    bpp 0.015111
    PSNR 21.7000 / MS-SSIM 0.7400
    LPIPS 0.3818 / DISTS 0.2727
    control delta L1 -0.0056

fixed-target control-aware FT600:
  full552 bpp 0.014978
  PSNR 21.2643 / MS-SSIM 0.7133
  LPIPS 0.3649 / DISTS 0.2597
  condition_l1 0.4014

fixed-target DISTS FT500:
  full552 bpp 0.014977
  PSNR 21.2877 / MS-SSIM 0.7151
  LPIPS 0.3651 / DISTS 0.2610
  condition_l1 0.4017

hybrid selector DISTS-RDO rate lambda 0.5:
  limit64 bpp 0.014955
  PSNR 21.5853 / MS-SSIM 0.7342
  LPIPS 0.3538 / DISTS 0.2544
  mode counts: affine_basis 45, affine 9, affine_dct 4, none 6
```

Decision:

```text
Do not promote these branches.

Base-fixed control improves condition after complement training but pushes the
output toward fidelity and away from LPIPS/DISTS. The corrected fixed-target
training improves PSNR/MS-SSIM/LPIPS but not DISTS. Hybrid selector RDO gives
only small metric-specific gains unless it spends far too many bits.

The active mainline remains the CoD_Lite_bpp_0_0156 denoiser-all LoRA + k16
sparse counted-control path. The next large-gain work should change the
CoSER information flow: diffusion-friendly detail/control heads, Stage4-aware
detail allocation, and better semantic/detail rate allocation. Do not spend
more time on base-fixed, loss-only DISTS reweighting, or selector-only tuning
until the detail/control representation improves.
```

See:

```text
docs/research/design_decisions/070_stage5_basefixed_fixedtarget_and_hybrid_selector_probe_20260629.md
```

## 2026-06-29 Prefix-TopK Entropy / Spatial Gate Update

New Stage 5 probes after the LoRA-all k16 anchor:

```text
prefix16 + tail topk16 entropy control:
  run 20260629_stage5_lora0156_all_lorabasis_prefix16topk16_huff_full552
  bpp 0.015403
  control_bpp 0.001404
  PSNR 21.1966 / MS-SSIM 0.7117
  LPIPS 0.3659 / DISTS 0.2572
  condition_l1 0.4026

k16 sparse control + spatial condition gate:
  run 20260629_stage5_lora0156_all_k16_spatial_gate_tv_full552
  bpp 0.014980
  control_bpp 0.000981
  PSNR 21.2206 / MS-SSIM 0.7127
  LPIPS 0.3679 / DISTS 0.2592
  condition_l1 0.4030
  post_control_gate_mean 0.9555
```

Decision:

```text
Do not promote either as the primary Stage 5 result.

prefix_topk entropy is useful codec infrastructure and nearly matches k32
control at lower bpp, but the improvement over k16 is tiny.

spatial gate is a legitimate condition-space gate, not RGB postprocessing, but
it only lands between raw k16 and global-gated k16. It does not dominate both.

The next mainline step is not more gate-only tuning. Train a stronger
control-aware adapter/detail-control path that learns with counted control in
the loop.
```

See:

```text
docs/research/design_decisions/068_stage5_prefix_topk_entropy_and_spatial_gate_probe_20260629.md
```

## 2026-06-29 Current Stage 5 Mainline Update

The active Stage 5 continuation is now the `CoD_Lite_bpp_0_0156`
denoiser-all LoRA branch plus efficient k16 counted condition-control.

Best current no-extra-bit Stage 5 adapter/control state:

```text
run:
  20260629_stage5_detailtarget_lora0156_denoiser_all_r4_balanced_ft900_full552_eval

actual_payload_bpp:
  0.013999

metrics:
  PSNR     21.2926
  MS-SSIM   0.7149
  LPIPS     0.3867
  DISTS     0.2686
  condition_l1 0.4113
```

Interpretation:

```text
This is a real no-extra-bit perceptual improvement over the previous Stage 4
adapter anchor, with a small fidelity cost. Use it as the active Stage 5
backbone-adapted continuation, not as an external-baseline win.
```

Most efficient current counted-control point:

```text
run:
  20260629_stage5_lora0156_all_oldbasis_full552_affinebasis_k16

actual_payload_bpp:
  0.014978

control_payload_bpp:
  0.000978

metrics:
  PSNR     21.1966
  MS-SSIM   0.7117
  LPIPS     0.3665
  DISTS     0.2575
  condition_l1 0.4030
```

Current balanced Stage 5 operating point:

```text
run:
  20260629_stage5_lora0156_all_k16_global_gate_perceptual_full552

actual_payload_bpp:
  0.014978

control_payload_bpp:
  0.000978

metrics:
  PSNR     21.2573
  MS-SSIM   0.7142
  LPIPS     0.3723
  DISTS     0.2622
  condition_l1 0.4037
  post_control_gate_mean 0.9085
```

Interpretation:

```text
The learned global condition-space gate slightly dominates the fixed
scale-0.5 diagnostic and gives a safer balanced point than full k16 control.
It is useful, but the gate distribution is still narrow/high, so the next
large gains should come from better adapter/control/detail representation,
not gate-only sweeps.
```

Rejected branch:

```text
CoD_Lite_bpp_0_0078 denoiser-all LoRA:
  run 20260629_stage5_detailtarget_lora0078_denoiser_all_r4_balanced_ft700_full552_eval
  bpp 0.013999
  PSNR 20.8498 / MS-SSIM 0.6885 / LPIPS 0.4001 / DISTS 0.2785
  condition_l1 0.4619
```

Interpretation:

```text
The lower-rate CoD-Lite backbone does not help the current CoSER condition
path. Continue from 0.0156; do not spend more on 0.0078 unless the condition
adapter/control representation changes substantially.
```

See:

```text
docs/research/design_decisions/067_stage5_lora_all_control_gate_rate_probe_20260629.md
```

## 2026-06-29 Late Active Update

The current no-extra-bit Stage 4 adapter anchor is now the long
detail-residual-target continuation:

```text
run:
  20260629_stage4_detailtarget_perceptual_long_ft1800_b8_full552_eval

actual_payload_bpp:
  0.013999

metrics:
  PSNR     21.3043
  MS-SSIM   0.7166
  LPIPS     0.4080
  DISTS     0.2780
  condition_l1 0.4104
```

This supersedes the earlier detail-contrast Stage 4 anchor for active
continuation. It is still not an external-baseline win.

The current strongest internal Stage 5 perceptual counted-control candidate is:

```text
run:
  20260629_stage5_detailtarget_perceptual_long_g32basis_postgate_controlaware_perceptual_full552...topk_c256_k32...

actual_payload_bpp:
  0.015484

control_payload_bpp:
  0.001485

metrics:
  PSNR     21.1337
  MS-SSIM   0.7101
  LPIPS     0.3696
  DISTS     0.2572
  condition_l1 0.4043
```

The current balanced Stage 5 counted-control anchor is:

```text
run:
  20260629_stage5_detailtarget_perceptual_long_g32basis_postgate_balanced_full552...topk_c256_k32...

actual_payload_bpp:
  0.015484

control_payload_bpp:
  0.001485

metrics:
  PSNR     21.2378
  MS-SSIM   0.7146
  LPIPS     0.3825
  DISTS     0.2660
  condition_l1 0.4013
  gate_mean    1.0046
```

Interpretation:

```text
The perceptual gate gives the strongest internal LPIPS/DISTS/FID pressure.
The balanced gate is the safer continuation anchor because it keeps much more
fidelity and slightly improves over fixed control scale 1.0 at unchanged
actual_payload_bpp.
```

Full552 feature-use ablations on the balanced Stage 5 anchor:

```text
semantic latent shuffle:
  PSNR -0.1395
  MS-SSIM -0.0076
  LPIPS +0.0083
  DISTS +0.0035
  condition_l1 +0.0053

detail context zero:
  PSNR +0.0400
  MS-SSIM -0.0001
  LPIPS +0.0111
  DISTS +0.0054
  condition_l1 +0.0062
```

Interpretation:

```text
The balanced Stage 5 path clearly uses semantic latent information.
Detail context helps perceptual/condition quality but is not yet a fidelity
booster, so the next mainline detail work should make the same transmitted
detail payload more diffusion-control-friendly.
```

Initial actual-bpp control curve with the same balanced gate:

```text
no-extra Stage4:
  bpp 0.013999
  PSNR 21.3043 / MS-SSIM 0.7166 / LPIPS 0.4080 / DISTS 0.2780

top-k16:
  bpp 0.014977
  control_bpp 0.000978
  PSNR 21.2341 / MS-SSIM 0.7144 / LPIPS 0.3833 / DISTS 0.2662

top-k24:
  bpp 0.015233
  control_bpp 0.001234
  PSNR 21.2351 / MS-SSIM 0.7144 / LPIPS 0.3828 / DISTS 0.2660

top-k32:
  bpp 0.015484
  control_bpp 0.001485
  PSNR 21.2378 / MS-SSIM 0.7146 / LPIPS 0.3825 / DISTS 0.2660

top-k48:
  bpp 0.015984
  control_bpp 0.001985
  PSNR 21.2392 / MS-SSIM 0.7145 / LPIPS 0.3820 / DISTS 0.2658
```

Interpretation:

```text
The first roughly 0.001 bpp of counted control gives most of the current
LPIPS/DISTS gain. More sparse coefficients show diminishing returns, so the
next large gains likely require better control/detail representation or
rate-specific adapter/gate training, not only larger top-k.
```

Top-k16-specific gate probe:

```text
run:
  20260629_stage5_detailtarget_long_topk16_specific_gate_full552

same actual bpp as top-k16 curve point:
  0.014977

delta versus reusing the top-k32-trained balanced gate:
  PSNR    -0.0127
  MS-SSIM -0.0005
  LPIPS   -0.0014
  DISTS   -0.0012
```

Interpretation:

```text
Rate-specific gate training can improve perceptual metrics at equal actual
bpp, but the first top-k16 run trades away a little fidelity. Continue with
rate-specific gates, but add stronger fidelity guards or multi-objective
selection.
```

Top-k16 gate regularization probe:

```text
smoothguard:
  bpp 0.014977
  PSNR 21.2424 / MS-SSIM 0.7150 / LPIPS 0.3866 / DISTS 0.2674

midguard:
  bpp 0.014977
  PSNR 21.2357 / MS-SSIM 0.7147 / LPIPS 0.3846 / DISTS 0.2666
```

Interpretation:

```text
Gate deviation/TV regularization is implemented and works as a stability
control, but it did not produce a top-k16 Pareto improvement. Stronger
smoothing recovers fidelity while weakening perceptual gains; weaker smoothing
mostly returns to the reused-gate point. Do not spend more effort on gate-only
sweeps until the adapter/control representation improves.
```

GenCodec-style patch-FID/KID versus Stage 3:

```text
Kodak24:
  Stage3 216.3567 / 0.150822
  Stage5  77.1709 / 0.019143
  balanced 81.1976 / 0.022353

CLIC2020 test428:
  Stage3 152.0035 / 0.079694
  Stage5  56.3215 / 0.006421
  balanced 57.6185 / 0.006986

DIV2K val100:
  Stage3 234.6295 / 0.092627
  Stage5 151.8839 / 0.012842
  balanced 153.6698 / 0.013911
```

Interpretation:

```text
Stage 5 now gives large internal perceptual/distribution improvements from the
CoSER semantic/detail actual bitstream plus a tiny counted control stream.
However, official CoD-Lite remains far ahead on Kodak LPIPS/DISTS/FID at a
comparable nominal rate, so do not claim external curve superiority.
```

See:

```text
docs/research/design_decisions/065_stage4_detailtarget_long_and_stage5_rebased_control_20260629.md
```

## 2026-06-29 Earlier Active Update

The current no-extra-bit Stage 4 adapter anchor is now the detail-contrast
CoD-Lite condition adapter:

```text
run:
  20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8_full552_eval

actual_payload_bpp:
  0.013999

metrics:
  PSNR     21.2351
  MS-SSIM   0.7166
  LPIPS     0.4139
  DISTS     0.2821
  condition_l1 0.4108
```

This is the fidelity-tilted no-extra-bit anchor. It is not an
external-baseline win.

The current no-extra-bit balanced condition-strength gate operating point is:

```text
run:
  20260629_stage4_condition_strength_gate_balanced_spatial_ft800_b2ga2_full552_eval

actual_payload_bpp:
  0.013999

metrics:
  PSNR     21.2121
  MS-SSIM   0.7154
  LPIPS     0.4068
  DISTS     0.2786
  condition_l1 0.4106
  condition_gate_mean 1.0488
```

This gate is deterministic from decoded CoSER features and fixed model weights.
It adds no image-specific side information. Versus the ungated detail-contrast
adapter, it improves LPIPS/DISTS and patch-FID/KID on every strict full552
split while slightly reducing PSNR/MS-SSIM.

The current Stage 5 counted-control interpretation is:

```text
balanced sparse counted-control anchor:
  20260629_stage5_detailcontrast_hf_g32basis_full552...sc1...topk_c256_k32...
  actual_payload_bpp 0.015486
  control_payload_bpp about 0.00149
  PSNR 21.1617 / MS-SSIM 0.7149 / LPIPS 0.3891 / DISTS 0.2691

perceptual sparse counted-control operating point:
  20260629_stage5_detailcontrast_hf_g32basis_full552...sc125...topk_c256_k32...
  actual_payload_bpp 0.015488
  control_payload_bpp about 0.00149
  PSNR 21.1319 / MS-SSIM 0.7140 / LPIPS 0.3844 / DISTS 0.2663

post-control deterministic gate perceptual-control probe:
  20260629_stage5_detailcontrast_hf_g32basis_postgate_balanced_full552...sc1...topk_c256_k32...
  actual_payload_bpp 0.015486
  control_payload_bpp about 0.00149
  PSNR 21.1249 / MS-SSIM 0.7129 / LPIPS 0.3813 / DISTS 0.2650
```

The counted-control stream is image-specific side information and is counted in
`actual_payload_bpp`. The current post-control deterministic gate improves
LPIPS/DISTS beyond the fixed `control_scale=1.25` point and improves
Kodak/CLIC patch-FID/KID, but it loses additional fidelity and slightly worsens
DIV2K patch-FID/KID. Treat it as a mainline probe for a stronger control-aware
gate, not as the balanced Stage 5 anchor.

Patch-FID/KID under the GenCodec-style patch protocol:

```text
Kodak:
  Stage3 FID/KID 216.3567 / 0.150822
  no-extra base  90.8343 / 0.029419
  balanced gate  87.9047 / 0.027304
  Stage5 sc1.25  79.2004 / 0.020942
  post-gate sc1  78.6804 / 0.020334

CLIC2020 test:
  Stage3 FID/KID 152.0035 / 0.079694
  no-extra base  61.3787 / 0.008463
  balanced gate  60.6337 / 0.008175
  Stage5 sc1.25  57.4915 / 0.007016
  post-gate sc1  57.4315 / 0.006960

DIV2K val:
  Stage3 FID/KID 234.6295 / 0.092627
  no-extra base  160.6237 / 0.016803
  balanced gate  159.1860 / 0.016311
  Stage5 sc1.25  152.6266 / 0.013602
  post-gate sc1  152.9545 / 0.013650
```

See:

```text
docs/research/design_decisions/061_stage4_detailcontrast_and_stage5_sparse_control.md
```

Tooling guardrails added on 2026-06-29:

```text
eval_stage4_cod_lite_adapter:
  missing actual_payload_bpp can no longer silently report 0 in evaluation.

fit_stage5_condition_control_basis:
  --huffman-quantile accepts p99 and 0.99 forms.
  stdout is compact; full Huffman tables remain in summary artifacts.

sweep_stage5_counted_control:
  default --limit is 0 for full-manifest behavior.
  fast screening should explicitly pass --limit 64.
```

See:

```text
docs/research/design_decisions/059_stage4_dists_adapter_and_counted_control_guardrails.md
docs/research/design_decisions/061_stage4_detailcontrast_and_stage5_sparse_control.md
```

## Bottom Line

The project is ready to start the Core-MVP Stage 4 diffusion integration from
the valid Stage 1-3 bitstream path.

This does not mean CoSER-DiC has already shown a large improvement over
published external methods. It means the internal codec substrate is now stable
enough to test the MVP's intended diffusion-backbone hypothesis:

```text
decoded semantic/detail actual bitstream
  -> auxiliary reconstruction and decoded feature controls
  -> CoSER adapter
  -> CoD-Lite default backbone or CoD parallel heavy backbone
  -> perceptual reconstruction
```

The ResUNet decoder-refiner experiments are archived diagnostics only. They
should not be used to choose the Stage 4 architecture.

Mainline policy:

```text
do not optimize fixed alpha blend as the method
do not promote learned RGB output alpha as the method
use alpha0.25/alpha0.30 and learned RGB output gates only as diagnostics
focus next work on condition recovery, deterministic gating, diffusion-friendly
detail features, training scale, rate allocation, and official baselines
```

See:

```text
docs/research/design_decisions/024_stage4_stage5_mainline_research_direction.md
docs/research/design_decisions/025_stage4_no_posthoc_rgb_mainline_condition_gate.md
docs/research/design_decisions/026_stage4_coser_feature_ablation_and_guarded_adapter.md
docs/research/design_decisions/027_stage4_detailaware_condition_adapter.md
docs/research/design_decisions/028_stage4_rate_specific_cod_lite_backbone_probe.md
docs/research/design_decisions/029_stage4_dists_loss_probe_and_traincache_scale.md
docs/research/design_decisions/030_stage4_fast_train_cache_and_8192_adapter_scale.md
docs/research/design_decisions/031_stage4_fast8192_coser_feature_ablation.md
docs/research/design_decisions/032_stage4_semantic_dropout_detail_use_probe.md
docs/research/design_decisions/033_stage4_detail_control_branch_probe.md
docs/research/design_decisions/034_stage4_detail_highfreq_residual_loss_probe.md
docs/research/design_decisions/035_stage4_condition_gate_dists_probe.md
docs/research/design_decisions/036_stage4_detail_film_condition_adapter.md
docs/research/design_decisions/037_bd_rate_tooling_and_current_external_gap.md
docs/research/design_decisions/038_stage5_partial_backbone_finetune_path.md
docs/research/design_decisions/039_stage5_tiny_counted_condition_control_stream.md
docs/research/design_decisions/040_stage5_lora_backbone_adaptation_path.md
docs/research/design_decisions/041_stage5_condition_residual_rms_guard_probe.md
docs/research/design_decisions/042_stage5_anchor_aware_candidate_selection.md
docs/research/design_decisions/043_gpu_preflight_for_long_experiments.md
docs/research/design_decisions/044_stage5_counted_affine_condition_control.md
docs/research/design_decisions/045_stage5_counted_affine_dct_condition_control.md
docs/research/design_decisions/046_stage5_counted_affine_grid_condition_control.md
docs/research/design_decisions/047_stage5_counted_affine_basis_condition_control.md
docs/research/design_decisions/048_stage5_counted_control_master_screen_selection.md
docs/research/design_decisions/049_stage5_counted_hybrid_condition_control.md
docs/research/design_decisions/050_stage5_counted_hybrid_basis_condition_control.md
docs/research/design_decisions/051_stage5_hybrid_image_rdo_selector.md
```

## Active Stage 1-3 Anchor

Use the batch-16 rate-prior Stage 1 branch as the current internal
rate-perception anchor:

```text
stage1_checkpoint:
  checkpoints/stage1_semantic_vq/20260627_stage1_lpips002_rateprior0005_tau01_vq005_freezecodebook_ft500_b16.pt

stage2_crop512_token_prior:
  checkpoints/stage2_token_prior/20260628_stage2_token_prior_crop512_tf384_l6_12kstep_32768tokens_stage1_rateprior0005_b16_lr1e4_do02_amp_es.pt

stage2_crop512_topk2048_entropy_prior:
  outputs/stage2_learned_entropy/20260628_stage2_learned_topk2048_escape_huffman_fit_32768tokens_crop512_stage1_rateprior0005_b16_tf384_l6_12k/learned_topk_escape_huffman_prior.json

stage3_crop512_detail_prior:
  outputs/stage3_residual_entropy/20260628_stage3_residual_semposleft_g4_sm0_d32_b4_r025_crop512_8192calib_stage1_rateprior0005_b16/static_residual_grid_semantic_position_leftctx_huffman_prior.json
```

This path is valid for strict 512-crop GenCodec-style evaluation because it has
a 16x16 semantic context, a matching 16x16 residual prior, decoder-prefix-safe
top-k entropy coding, and exact semantic/detail roundtrip.

## Strict 512 Stage 3 Reference Result

Canonical strict 512 GenCodec-split result:

```text
dataset:
  Kodak 24
  CLIC2020 test 428
  DIV2K val 100
  total: 552 images
  crop: 512 x 512 center crop

run:
  20260628_stage3_b16_topk2048_tf384_semposleft_g4_b4_gencodec512_pp_unsharp200_perceptual_prefixsafe

actual_payload_bpp / paper_bpp:
  0.013999

semantic/detail bpp:
  0.008997 / 0.005002

roundtrip:
  semantic=true
  detail=true
  failure_count=0

PSNR / MS-SSIM:
  21.989628 / 0.734834

LPIPS / DISTS:
  0.576842 / 0.353785
```

Do not use the earlier teacher-forced top-k 512 run that had semantic
roundtrip failures.

## Stage 3 Operating Points

For Stage 4 conditioning, prefer the actual decoded fields before cosmetic
postprocess as the clean auxiliary input. Fixed gain/unsharp settings are useful
as Stage 3 reporting operating points, not as the main Stage 4 design axis.

Current Stage 3 reporting interpretation:

```text
no-postprocess:
  safest PSNR / neutral auxiliary reconstruction point

gain1.00 + unsharp0.20:
  DISTS-first Stage 3 point

gain0.625 + unsharp0.15:
  balanced LPIPS/FID point

gain0.50 + unsharp0.15:
  strongest checked LPIPS and CoD-style patch-FID point
  sacrifices PSNR and DISTS
```

These settings are fixed decoder-side operating choices. They do not add
image-specific payload bpp. If a future method transmits image-specific gain,
mask, prompt, seed, map, or control information, it must be counted in
`actual_payload_bpp`.

## Refiner Diagnostic Status

The ResUNet-style decoder refiner is not part of the active MVP Stage 4 path.

Keep from it:

```text
validated:
  actual-bpp-preserving decoder-side evaluation scaffold
  conditioning I/O smoke tests
  need for full Kodak / CLIC2020 / DIV2K distribution gates

not validated:
  CoD or CoD-Lite diffusion behavior
  adapter design for official GenCodec backbones
  ability to beat CoD / CoD-Lite baselines
```

Therefore, do not infer that a ResUNet gain or failure will transfer to CoD or
CoD-Lite.

## Active Stage 4 Plan

Default CoD-Lite track:

```text
CoD-Lite backbone candidates:
  external/pretrained/CoD_Lite/CoD_Lite_pretrain.pt
  external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0078.pt
  external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0156.pt
  external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0312.pt
```

Current Stage 4 anchors are already based on rate-specific CoD-Lite
checkpoints, not only the generic pretrain checkpoint:

```text
bpp0.0156:
  current LPIPS / patch-FID raw anchor

bpp0.0312:
  current DISTS / fidelity raw anchor

bpp0.0078:
  first transfer probe not promoted
```

Parallel heavy track:

```text
CoD checkpoints:
  external/pretrained/CoD/cod/CoD_pixel_vpred.pt
  external/pretrained/CoD/cod/CoD_latent_vpred.pt
  external/pretrained/CoD/cod/CoD_latent_vpred_64bits.pt
  external/pretrained/CoD/finetuned_one_step_cod/bpp_0_0039.pt
  external/pretrained/CoD/finetuned_one_step_cod/bpp_0_0312.pt
  external/pretrained/CoD/finetuned_one_step_cod/bpp_0_1250.pt
```

The implementation should wrap the official GenCodec/CoD-Lite backbones as
fixed or low-lr decoder components, then train a CoSER-owned conditioning
adapter from scratch. The adapter must consume only decoded information unless
any extra image-specific side information is explicitly entropy-coded and
counted.

First Stage 4 comparisons:

```text
Stage 3 x_aux only
CoD-Lite official rate checkpoint baseline
CoD official / one-step baseline where reproducible
CoSER + frozen CoD-Lite backbone + adapter
CoSER + frozen CoD backbone + adapter
```

Promotion gate:

```text
actual_payload_bpp remains Stage 3 semantic/detail payload bpp
roundtrip failure_count = 0
LPIPS / DISTS improve over Stage 3 at the same payload
FID/KID do not regress under the selected CoD/CoD-Lite protocol
visual failures are audited, especially structured high-frequency content
```

## Current Stage 4 Detail-Control Diagnosis

Recent mainline probes tried to make the decoded detail payload more
consequential for CoD-Lite condition recovery:

```text
semantic-latent dropout:
  improves LPIPS/DISTS slightly, but does not materially increase detail use

direct detail-control branch:
  improves condition_l1, but detail zero/shuffle sensitivity remains mild

detail high-frequency residual loss:
  increases detail ablation sensitivity, but hurts LPIPS/DISTS on limit64

DISTS-aware condition-space gate:
  slightly recovers PSNR/MS-SSIM versus raw adapter, but worsens LPIPS/DISTS
  and collapses close to the raw adapter branch

detail-FiLM condition adapter:
  small full552 improvement over semantic-dropout detail-control in PSNR,
  LPIPS, DISTS, and condition L1 at unchanged actual payload bpp; MS-SSIM
  slightly regresses

BD-rate readiness:
  current CoSER Kodak LPIPS/DISTS do not overlap official CoD-Lite Kodak512
  curve range, so no LPIPS/DISTS BD-rate can be claimed yet
```

Current implication:

```text
detail stream is not yet a strong enough diffusion-control signal
global condition L1 / high-frequency residual losses are too blunt
gating an imperfect residual is not enough
detail-FiLM is a modestly better diffusion-control detail head
but Stage 5 needs a much larger perceptual jump, at least to Kodak512 LPIPS
below 0.339 and DISTS below 0.2089 before BD-rate against the stored official
CoD-Lite Kodak curve can even be computed
next attempts should extend this to rate-specific bpp0.0312 transfer, better
channel targets, group-wise residual control, partial unfreeze/LoRA, or a tiny
counted control stream instead of optimizing final RGB blends
```

## Active Stage 5 Next Step: Partial CoD-Lite Backbone Adaptation

Decision:

```text
docs/research/design_decisions/038_stage5_partial_backbone_finetune_path.md
```

Implementation status:

```text
scripts/train_stage4_cod_lite_adapter.py can now unfreeze selected official
CoD-Lite backbone parameters by regex with a separate low LR

checkpoints save only selected backbone_trainable_state tensors, plus the
CoSER adapter

evaluation / condition-stat / gate scripts load backbone_trainable_state before
decoding so Stage 5 candidates are evaluated with the same decoder state used
in training
```

Research interpretation:

```text
this is the next mainline move after adapter-only improvements saturated
it keeps semantic/detail CoSER streams as the only transmitted payload
it does not count decoder weight updates in actual_payload_bpp
it is not an RGB blend, postprocess, or replacement with a native CoD-Lite
bitstream
```

GPU status:

```text
NVML is currently unavailable and torch.cuda.is_available is false
do not launch the documented partial-backbone GPU command until the container
is restarted
generated long-running shell plans now call scripts/check_gpu_ready.py before
training/evaluation and should fail fast while this condition persists
```

## Active Stage 5 Parallel Step: Tiny Counted Control Stream

Decision:

```text
docs/research/design_decisions/039_stage5_tiny_counted_condition_control_stream.md
```

Implementation status:

```text
src/coserdic/entropy/control_grid.py adds UniformControlGridCode
scripts/eval_stage4_cod_lite_adapter.py can now evaluate a counted
condition_residual_grid, condition_residual_dct, or condition_residual_basis
control stream
scripts/sweep_stage5_counted_control.py can dry-run or execute the first
limit64 control sweep and records exact control bpp / summary paths; it can
also consume inspected basis-setting JSON directly to avoid manual p95/p99
range transcription, filter low-explanation / high-quantization-error settings
before limit64, and sort candidates by retained energy per planned bpp
scripts/fit_stage5_condition_control_basis.py can fit a fixed PCA/SVD basis
from train-cache condition residuals after GPU restart and save static
position-conditioned Huffman priors for uniform or mu-law quantized basis
coefficients
scripts/inspect_stage5_control_basis.py can convert basis coefficient p95/p99
statistics into reproducible fixed/Huffman and uniform/mu-law basis sweep
settings with planned added bpp; it can now emit prefix and sparse top-k basis
selection settings and attach retained/residual condition-residual coefficient
energy diagnostics
scripts/plan_stage5_control_curve.py can select multi-rate control settings
from inspected control-bpp bands before limit64 so Stage 5 moves toward
BD-rate curves instead of a single operating point
scripts/select_stage5_control_candidates.py can rank executed limit64 control
sweeps with actual_payload_bpp, Pareto flags, Stage3 fidelity guards, and basis
retained/residual energy guards before full552 promotion; it can also attach
same-bpp and single-point proxy gaps to official CoD-Lite curves, clearly
labeled as not BD-rate
scripts/promote_stage5_control_candidates.py can rewrite selected limit64 eval
commands into full552 evaluation commands without changing the selected control
settings by hand
scripts/collect_bd_curve_points.py can now collect promoted full552 summary
paths from the promotion JSON and build an actual_payload_bpp curve JSON
it can also merge separately computed patch-FID/KID JSON metrics into labeled
curve points
scripts/compute_bd_rate.py emits dataset / bpp-policy mismatch warnings and can
fail strict promotion checks with --fail-on-dataset-mismatch
actual_payload_bpp / paper_bpp is Stage 3 bpp plus control_payload_bpp
```

Research interpretation:

```text
this is a legitimate Stage 5 information-flow route, not RGB postprocessing
the encoder may use the original image to produce control bytes
the decoder only receives decoded CoSER semantic/detail tensors, fixed model
state, and decoded control bytes
for basis mode, the fitted basis/mean are fixed decoder-side model state, while
only per-image coefficients are counted as payload
for Huffman basis mode, the code lengths are fixed decoder-side model state and
the per-image Huffman payload bytes are counted exactly
for mu-law basis mode, the compander parameter is fixed decoder-side model
state and no per-image side information is hidden
for sparse top-k basis mode, both the selected coefficient index stream and
the selected coefficient value stream are image-specific payload and must be
counted exactly
for sparse top-k Huffman mode, the index/value code lengths are fixed
decoder-side model state fitted from non-eval train-cache selections, while
the per-image index/value Huffman payload bytes are counted exactly
the control stream must always be reported as extra actual payload bpp
```

## Active Stage 5 Counted Affine Condition-Control Probe

Decision:

```text
docs/research/design_decisions/044_stage5_counted_affine_condition_control.md
```

Implementation status:

```text
scripts/eval_stage4_cod_lite_adapter.py supports:
  --counted-control-mode condition_residual_affine

scripts/sweep_stage5_counted_control.py can generate affine control settings
and write executable shell plans with GPU preflight

scripts/select_stage5_control_candidates.py preserves affine control metadata:
  counted_control_mode
  control_grid_size
  control_groups
  control_affine_gain_range
  control_affine_bias_range
```

Generated affine limit64 plan:

```text
results/stage5_counted_control/20260629_detailfilm_affine_limit64_sweep_plan.json
results/stage5_counted_control/20260629_detailfilm_affine_limit64_sweep_plan.sh
```

Curve-oriented six-candidate screen generated from the same affine plan:

```text
results/stage5_counted_control/20260629_detailfilm_affine_limit64_curve_settings.json
results/stage5_counted_control/20260629_detailfilm_affine_curve_limit64_sweep_plan.json
results/stage5_counted_control/20260629_detailfilm_affine_curve_limit64_sweep_plan.sh
```

This plan keeps two candidates per extra-control-bpp band:

```text
0.0000-0.0005
0.0005-0.0011
0.0011-0.0021
```

Implementation hygiene:

```text
scripts/sweep_stage5_counted_control.py now round-trips string and dict
setting rows, supports --include-mode, and strips stale command/summary/run_name
metadata when replanning from a prior JSON plan

scripts/plan_stage5_control_curve.py supports --include-mode, so affine,
grid, DCT, and basis settings can be planned through the same curve-selection
pipeline
```

Planned extra actual payload:

```text
min control bpp:
  0.000244140625

max control bpp:
  0.001953125
```

Research interpretation:

```text
this is a mainline Stage 5 information-flow change
it sends per-image quantized gain/bias controls for the CoD-Lite condition
residual before decoding
it is not final RGB blending or uncounted side information
the hypothesis is that affine correction can fix adapter scale/bias mismatch
more efficiently than additive residual grids at the same tiny payload
```

After GPU restart:

```bash
bash results/stage5_counted_control/20260629_detailfilm_affine_curve_limit64_sweep_plan.sh
```

Then select against the matching limit64 detail-FiLM anchor:

```bash
.venv/bin/python scripts/select_stage5_control_candidates.py \
  --sweep-plan results/stage5_counted_control/20260629_detailfilm_affine_curve_limit64_sweep_plan.json \
  --anchor-summary results/stage4_cod_lite_adapter_eval/20260629_stage4_fast8192_detailfilm_ft800_b8_limit64_eval/summary.json \
  --fail-on-anchor-count-mismatch \
  --max-psnr-drop 0.8 \
  --max-anchor-psnr-drop 0.25 \
  --max-anchor-ms-ssim-drop 0.01 \
  --max-anchor-lpips-increase 0.0 \
  --max-anchor-dists-increase 0.0 \
  --output-json results/stage5_selection/20260629_detailfilm_affine_limit64_anchor_selection.json
```

## Active Stage 5 Counted Affine+DCT Condition-Control Probe

Decision:

```text
docs/research/design_decisions/045_stage5_counted_affine_dct_condition_control.md
```

Implementation status:

```text
scripts/eval_stage4_cod_lite_adapter.py supports:
  --counted-control-mode condition_residual_affine_dct

The mode applies:
  affine condition correction
  then grouped DCT correction on the remaining condition error

scripts/sweep_stage5_counted_control.py can plan affine_dct settings and counts:
  affine gain bytes
  affine bias bytes
  DCT residual bytes
```

Generated pending limit64 plan:

```text
results/stage5_counted_control/20260629_detailfilm_affinedct_limit64_sweep_plan.json
results/stage5_counted_control/20260629_detailfilm_affinedct_limit64_sweep_plan.sh
```

Planned extra actual payload:

```text
min control bpp:
  0.00048828125

max control bpp:
  0.00146484375
```

Research interpretation:

```text
affine-only corrects condition residual scale/bias
affine_dct additionally spends a few counted bits on low-frequency remaining
condition error
this is still condition-space decoding control, not RGB postprocessing
```

After GPU restart:

```bash
bash results/stage5_counted_control/20260629_detailfilm_affinedct_limit64_sweep_plan.sh
```

Then select against the matching limit64 detail-FiLM anchor:

```bash
.venv/bin/python scripts/select_stage5_control_candidates.py \
  --sweep-plan results/stage5_counted_control/20260629_detailfilm_affinedct_limit64_sweep_plan.json \
  --anchor-summary results/stage4_cod_lite_adapter_eval/20260629_stage4_fast8192_detailfilm_ft800_b8_limit64_eval/summary.json \
  --fail-on-anchor-count-mismatch \
  --max-psnr-drop 0.8 \
  --max-anchor-psnr-drop 0.25 \
  --max-anchor-ms-ssim-drop 0.01 \
  --max-anchor-lpips-increase 0.0 \
  --max-anchor-dists-increase 0.0 \
  --output-json results/stage5_selection/20260629_detailfilm_affinedct_limit64_anchor_selection.json
```

## Active Stage 5 Counted Affine+Grid Condition-Control Probe

Decision:

```text
docs/research/design_decisions/046_stage5_counted_affine_grid_condition_control.md
```

Implementation status:

```text
scripts/eval_stage4_cod_lite_adapter.py supports:
  --counted-control-mode condition_residual_affine_grid

The mode applies:
  affine condition correction
  then grouped grid correction on the remaining condition error

scripts/sweep_stage5_counted_control.py can plan affine_grid settings and counts:
  affine gain bytes
  affine bias bytes
  grid residual bytes
```

Generated pending limit64 plan:

```text
results/stage5_counted_control/20260629_detailfilm_affinegrid_limit64_sweep_plan.json
results/stage5_counted_control/20260629_detailfilm_affinegrid_limit64_sweep_plan.sh
```

Planned extra actual payload:

```text
min control bpp:
  0.0003662109375

max control bpp:
  0.002197265625
```

Research interpretation:

```text
affine_grid is a higher-flexibility sibling of affine_dct
it may spend slightly more payload but can represent grouped spatial condition
residuals that are not well captured by the first few DCT coefficients
this is still condition-space decoding control, not RGB postprocessing
```

After GPU restart:

```bash
bash results/stage5_counted_control/20260629_detailfilm_affinegrid_limit64_sweep_plan.sh
```

Then select against the matching limit64 detail-FiLM anchor:

```bash
.venv/bin/python scripts/select_stage5_control_candidates.py \
  --sweep-plan results/stage5_counted_control/20260629_detailfilm_affinegrid_limit64_sweep_plan.json \
  --anchor-summary results/stage4_cod_lite_adapter_eval/20260629_stage4_fast8192_detailfilm_ft800_b8_limit64_eval/summary.json \
  --fail-on-anchor-count-mismatch \
  --max-psnr-drop 0.8 \
  --max-anchor-psnr-drop 0.25 \
  --max-anchor-ms-ssim-drop 0.01 \
  --max-anchor-lpips-increase 0.0 \
  --max-anchor-dists-increase 0.0 \
  --output-json results/stage5_selection/20260629_detailfilm_affinegrid_limit64_anchor_selection.json
```

## Active Stage 5 Counted Affine+Basis Condition-Control Probe

Decision:

```text
docs/research/design_decisions/047_stage5_counted_affine_basis_condition_control.md
```

Implementation status:

```text
scripts/eval_stage4_cod_lite_adapter.py supports:
  --counted-control-mode condition_residual_affine_basis

scripts/fit_stage5_condition_control_basis.py supports:
  --pre-basis-affine true

scripts/sweep_stage5_counted_control.py supports:
  mode=affine_basis
  --override-mode affine_basis
```

Research interpretation:

```text
affine_basis first removes cheap scale/bias mismatch, then spends basis
coefficients on the remaining condition error
this should be more bit-efficient than raw basis control if affine absorbs
dominant low-dimensional error components
this is still condition-space decoding control, not RGB postprocessing
```

GPU-restart prepare entrypoint:

```bash
bash results/stage5_counted_control/20260629_detailfilm_postaffine_basis_prepare_plan.sh
```

The prepare plan will generate:

```text
outputs/stage5_control_basis/20260629_detailfilm_postaffine_basis_g16s8_k64_train8192/control_basis.pt
outputs/stage5_control_basis/20260629_detailfilm_postaffine_basis_g16s8_k64_train8192/recommended_basis_settings.json
results/stage5_counted_control/20260629_detailfilm_affinebasis_limit64_sweep_plan.json
results/stage5_counted_control/20260629_detailfilm_affinebasis_limit64_sweep_plan.sh
```

Then run:

```bash
bash results/stage5_counted_control/20260629_detailfilm_affinebasis_limit64_sweep_plan.sh
```

Then select against the matching limit64 detail-FiLM anchor:

```bash
.venv/bin/python scripts/select_stage5_control_candidates.py \
  --sweep-plan results/stage5_counted_control/20260629_detailfilm_affinebasis_limit64_sweep_plan.json \
  --anchor-summary results/stage4_cod_lite_adapter_eval/20260629_stage4_fast8192_detailfilm_ft800_b8_limit64_eval/summary.json \
  --fail-on-anchor-count-mismatch \
  --max-psnr-drop 0.8 \
  --max-anchor-psnr-drop 0.25 \
  --max-anchor-ms-ssim-drop 0.01 \
  --max-anchor-lpips-increase 0.0 \
  --max-anchor-dists-increase 0.0 \
  --output-json results/stage5_selection/20260629_detailfilm_affinebasis_limit64_anchor_selection.json
```

## Active Stage 5 Counted-Control Master Screen

Decision:

```text
docs/research/design_decisions/048_stage5_counted_control_master_screen_selection.md
```

Preferred GPU-restart entrypoint for the current counted-control family:

```bash
bash results/stage5_counted_control/20260629_stage5_mainline_screen_select_plan.sh
```

This master plan runs, in order:

```text
affine curve-oriented limit64 screen
affine+DCT limit64 screen
affine+grid limit64 screen
hybrid none/affine/affine+DCT/affine+grid limit64 screen
image-RDO hybrid none/affine/affine+DCT/affine+grid limit64 screen
post-affine basis fit on the non-eval train cache
generated affine+basis limit64 screen
generated hybrid-basis limit64 screen
generated DISTS-RDO hybrid-basis limit64 screen
anchor-aware selection across all executed counted-control families
full552 promotion-plan generation for the selected candidates
```

It starts with the GPU preflight:

```bash
.venv/bin/python scripts/check_gpu_ready.py --min-devices 1
```

so it should not be launched again until CUDA/NVML is visible.

Expected outputs after successful execution:

```text
results/stage5_selection/20260629_stage5_mainline_limit64_all_control_selection.json
results/stage5_selection/20260629_stage5_mainline_full552_promotion_plan.json
results/stage5_selection/20260629_stage5_mainline_full552_promotion_plan.sh
```

The master plan intentionally does not execute the generated full552 promotion
shell. Inspect the limit64 selection JSON first, then run:

```bash
bash results/stage5_selection/20260629_stage5_mainline_full552_promotion_plan.sh
```

This keeps Stage 5 promotion tied to one guarded selection artifact instead of
hand-picking a single control family.

## Active Stage 5 Counted Hybrid Condition-Control Probe

Decision:

```text
docs/research/design_decisions/049_stage5_counted_hybrid_condition_control.md
```

Implementation status:

```text
scripts/eval_stage4_cod_lite_adapter.py supports:
  --counted-control-mode condition_residual_hybrid_affine_dct_grid
  --control-hybrid-selector-bytes
  --control-hybrid-rate-lambda

scripts/sweep_stage5_counted_control.py supports:
  mode=hybrid_affine_dct_grid
  selector_bytes=<int>
  rd_lambda=<float>
```

Research interpretation:

```text
the encoder chooses none / affine / affine+DCT / affine+grid per image
the mode selector is image-specific and counted in actual_payload_bpp
selection uses a condition-space RD proxy, not final RGB postprocessing
the summary records control_hybrid_mode_counts so we can reject hybrid if it
collapses to one fixed family
```

Generated limit64 plan:

```text
results/stage5_counted_control/20260629_detailfilm_hybrid_limit64_sweep_plan.json
results/stage5_counted_control/20260629_detailfilm_hybrid_limit64_sweep_plan.sh
```

Planned conservative control bpp range:

```text
0.000762939453125 to 0.002227783203125 at 512x512
```

Image-RDO hybrid plan:

```text
results/stage5_counted_control/20260629_detailfilm_hybrid_imagerdo_limit64_sweep_plan.json
results/stage5_counted_control/20260629_detailfilm_hybrid_imagerdo_limit64_sweep_plan.sh
```

Objectives included:

```text
image_l1
image_mse
lpips_alex
dists
```

## Active Stage 5 Counted Hybrid-Basis Condition-Control Probe

Decision:

```text
docs/research/design_decisions/050_stage5_counted_hybrid_basis_condition_control.md
```

Implementation status:

```text
scripts/eval_stage4_cod_lite_adapter.py supports:
  --counted-control-mode condition_residual_hybrid_affine_dct_grid_basis

scripts/sweep_stage5_counted_control.py supports:
  mode=hybrid_affine_dct_grid_basis
  --override-mode hybrid_affine_dct_grid_basis
```

Research interpretation:

```text
after post-affine basis fitting, the encoder can choose per image among
none / affine / affine+DCT / affine+grid / affine+basis
the selector and selected payload bytes are counted
huffman/mu-law basis settings can be reused through the inspected basis JSON
promotion still depends on image metrics and anchor guards, not condition L1
```

Generated after basis prepare:

```text
results/stage5_counted_control/20260629_detailfilm_hybridbasis_limit64_sweep_plan.json
results/stage5_counted_control/20260629_detailfilm_hybridbasis_limit64_sweep_plan.sh
```

DISTS-RDO hybrid-basis plan generated after basis prepare:

```text
results/stage5_counted_control/20260629_detailfilm_hybridbasis_distsrdo_limit64_sweep_plan.json
results/stage5_counted_control/20260629_detailfilm_hybridbasis_distsrdo_limit64_sweep_plan.sh
```

## Active Stage 5 Hybrid Image-RDO Selector

Decision:

```text
docs/research/design_decisions/051_stage5_hybrid_image_rdo_selector.md
```

Implementation status:

```text
scripts/eval_stage4_cod_lite_adapter.py supports:
  --control-hybrid-selection-objective condition_l1
  --control-hybrid-selection-objective image_l1
  --control-hybrid-selection-objective image_mse
  --control-hybrid-selection-objective lpips_alex
  --control-hybrid-selection-objective dists

scripts/sweep_stage5_counted_control.py supports:
  objective=<condition_l1|image_l1|image_mse|lpips_alex|dists>
```

Research interpretation:

```text
image-RDO runs each candidate selected-control path through the decoder at
encode time and transmits only the selected selector + payload
this is allowed codec RDO, not decoder leakage
unselected payloads are not counted because they are not transmitted
selected payload bytes are still counted in actual_payload_bpp
```

## Active Stage 5 Parallel Step: LoRA Backbone Adaptation

Decision:

```text
docs/research/design_decisions/040_stage5_lora_backbone_adaptation_path.md
```

Implementation status:

```text
src/coserdic/models/gencodec_backbone.py adds LoRALinear / LoRAConv2d and
name-based LoRA insertion for official CoD-Lite Linear/Conv2d modules
scripts/train_stage4_cod_lite_adapter.py can train LoRA backbone adapters with
separate --backbone-lora-lr while keeping the pretrained backbone base weights
frozen
scripts/inspect_cod_lite_backbone_params.py can now inspect official CoD-Lite
checkpoints on CPU and report LoRA candidate module names and rank-specific
LoRA parameter counts
scripts/plan_stage5_lora_targets.py generated a bpp0.0312 rank-4 target plan
from the CPU inspection JSON
scripts/sweep_stage5_lora_backbone.py generated train + limit64 eval commands
for the first denoiser_tail and dec_net LoRA probes
scripts/train_stage4_cod_lite_adapter.py now supports optional training-time
condition residual RMS regularization for the same stability axis as the
condition residual guard probe
Stage 4 eval, condition-stat analysis, condition-gate training, RGB-gate
training, and Stage 5 control-basis fitting reconstruct LoRA modules from the
checkpoint before loading backbone_trainable_state
```

Research interpretation:

```text
LoRA is a mainline Stage 5 route because it adapts the CoD-Lite decoder prior
to CoSER condition tensors without adding image-specific payload or blending
final RGB outputs
LoRA weights are fixed decoder model state and are not counted in
actual_payload_bpp
the next GPU run should inspect parameter names first, then train a conservative
rank-4 LoRA continuation from the bpp0.0312 detail-aware/detail-FiLM anchor
```

Current bpp0.0312 LoRA target plan:

```text
inspect:
  results/stage5_inspect/20260629_cod_lite_bpp0312_lora_candidate_params_cpu.json

plan:
  results/stage5_inspect/20260629_cod_lite_bpp0312_lora_target_plan_rank4.json

sweep:
  results/stage5_lora/20260629_bpp0312_lora_denoiser_tail_decnet_limit64_plan.json
  results/stage5_lora/20260629_bpp0312_lora_denoiser_tail_decnet_limit64_plan.sh

current detail-FiLM guarded sweep:
  results/stage5_lora/20260629_detailfilm_lora_guarded_denoiser_tail_decnet_limit64_plan.json
  results/stage5_lora/20260629_detailfilm_lora_guarded_denoiser_tail_decnet_limit64_plan.sh
  condition_residual_rms_guard_weight: 0.05
  condition_residual_rms_guard_ratio: 0.35
  condition_residual_rms_guard_granularity: channel

denoiser_tail:
  30 modules / 147,456 rank-4 LoRA params
  first recommended GPU probe

dec_net:
  11 modules / 37,632 rank-4 LoRA params
  cheap stability probe

denoiser_all:
  105 modules / 516,096 rank-4 LoRA params

y_decoder:
  16 modules / 138,240 rank-4 LoRA params
  riskier, because it touches condition-codec decoder
```

## Active Stage 5 Stability Probe: Condition Residual RMS Guard

Decision:

```text
docs/research/design_decisions/041_stage5_condition_residual_rms_guard_probe.md
```

Implementation status:

```text
src/coserdic/models/gencodec_backbone.py adds condition_residual_rms_guard
scripts/eval_stage4_cod_lite_adapter.py can apply and log a deterministic
condition-space RMS guard before CoD-Lite decoding
scripts/sweep_stage5_condition_residual_guard.py generated screen64 and
full552 command plans for the current detail-FiLM adapter anchor
scripts/select_stage5_control_candidates.py preserves condition guard metadata
from sweep plans so guard candidates can be ranked with the same promotion
logic as LoRA/control candidates
```

Generated guard plans:

```text
screen64:
  results/stage5_condition_guard/20260629_detailfilm_guard_screen64_plan.json
  results/stage5_condition_guard/20260629_detailfilm_guard_screen64_plan.sh

full552:
  results/stage5_condition_guard/20260629_detailfilm_guard_full552_plan.json
  results/stage5_condition_guard/20260629_detailfilm_guard_full552_plan.sh
```

Research interpretation:

```text
this is not a Stage 5 win and not an RGB output blend
it is a condition-space stability probe for the current adapter residual
it adds no image-specific payload and therefore does not change
actual_payload_bpp / paper_bpp
run screen64 first after GPU restart, then promote only settings that beat the
guard_none branch without fidelity collapse
```

## Active Stage 5 Candidate Selection Rule

Decision:

```text
docs/research/design_decisions/042_stage5_anchor_aware_candidate_selection.md
```

Current internal anchor:

```text
20260629_stage4_fast8192_detailfilm_ft800_b8_full552_eval

actual_payload_bpp:
  0.013999109682829483

PSNR / MS-SSIM:
  21.2427 / 0.7150

LPIPS / DISTS:
  0.4304 / 0.2982
```

Implementation status:

```text
scripts/select_stage5_control_candidates.py now supports --anchor-summary and
--anchor-label

candidate JSON rows include *_delta_vs_anchor and *_improves_vs_anchor fields

promotion_guard_pass now also requires anchor_guard_pass when an anchor is
provided

generated Stage 5 and official baseline shell plans run scripts/check_gpu_ready.py
before launching GPU work
```

CPU analysis:

```text
results/stage5_selection/20260629_current_full552_vs_detailfilm_anchor_psnr08.json

loaded candidates:
  6 existing full552 Stage 4 branches

recommended:
  only the detail-FiLM anchor itself
```

Interpretation:

```text
existing Stage 4 branches do not beat the current detail-FiLM anchor under a
strict LPIPS/DISTS anchor guard
future Stage 5 guard / LoRA / counted-control candidates must be selected
against this anchor before full552 promotion
```

GPU-restart order:

```text
1. first run the counted-control master screen/select plan:
   results/stage5_counted_control/20260629_stage5_mainline_screen_select_plan.sh
2. inspect:
   results/stage5_selection/20260629_stage5_mainline_limit64_all_control_selection.json
3. only if recommended candidates pass the anchor guards, run:
   results/stage5_selection/20260629_stage5_mainline_full552_promotion_plan.sh
4. run the condition residual guard screen64 probe only if the selected
   candidates show instability that needs diagnosis
5. run the current detail-FiLM guarded LoRA limit64 plan from decision 040 as a
   parallel no-extra-payload route, not as a substitute for counted-control
   evaluation
6. collect multi-rate points only after protocol-matched full552 runs complete
7. compute BD-rate only when the reference and candidate datasets / bpp policy
   match
```

## Stage 4 Bootstrap Status

CoD-Lite adapter bootstrap has started:

```text
implementation:
  src/coserdic/models/gencodec_backbone.py
  scripts/train_stage4_cod_lite_adapter.py

decision:
  docs/research/design_decisions/022_stage4_cod_lite_adapter_bootstrap.md

status:
  official CoD-Lite one-step checkpoint loads through a CoSER wrapper
  frozen CoD-Lite backbone can decode from CoSER-produced condition tensors
  native-stage3 base condition is used so zero-init starts from decoded x_aux
  512-crop adapter training runs at batch size 1 on the local GPU
  LPIPS loss path runs without immediate NaNs

not promoted:
  raw adapter outputs do not yet beat external baselines
  visual samples still show generative structure distortion
```

Updated active track:

```text
decision:
  docs/research/design_decisions/023_stage4_semantic_latent_cod_lite_adapter.md

implementation:
  Stage 3 can now export decoder_feature_cache per image
  Stage 4 adapter can consume decoded 256-channel semantic latent tensors
  Stage 4 adapter can optionally consume decoded detail context
    residual_grid_hat
    normalized detail_codes
  condition residuals can be tanh-limited and gradient-clipped
  Stage 4 training supports gradient accumulation for larger effective batch

payload policy:
  semantic_latent is decoded from transmitted CoSER semantic tokens
  residual_grid_hat and detail_codes are decoded from transmitted CoSER detail tokens
  adapter and fixed CoD-Lite weights are decoder-side parameters
  no extra image-specific side information is counted unless introduced later
```

Kodak24 semantic-latent probe:

```text
stage3 feature cache:
  20260628_stage3_gencodec512_kodak24_feature_cache_smoke

train:
  20260628_stage4_cod_lite_pyramid_sem256_tanh075_kodak24_probe1k

eval:
  20260628_stage4_cod_lite_pyramid_sem256_tanh075_kodak24_probe1k_eval

actual_payload_bpp:
  0.014123

Stage 3:
  PSNR / MS-SSIM: 21.6674 / 0.7225
  LPIPS / DISTS: 0.6366 / 0.3732

Stage 4 raw:
  PSNR / MS-SSIM: 20.6458 / 0.6911
  LPIPS / DISTS: 0.4294 / 0.2743
```

Semantic latent input is better than RGB semantic input for the CoD-Lite
condition adapter, but the result still trails official CoD-Lite at comparable
rate. The next mainline run should train the semantic-latent + detail-context
adapter on non-evaluation train-cache exports, then evaluate on the strict
552-image CoD 512 reproduction split.

Strict full552 semantic-latent generalization:

```text
feature cache:
  20260628_stage3_gencodec512_full552_feature_cache

raw eval:
  20260628_stage4_cod_lite_pyramid_sem256_tanh075_kodak24_probe1k_full552_eval

alpha0.2 eval:
  20260628_stage4_cod_lite_pyramid_sem256_tanh075_kodak24_probe1k_full552_alpha020_eval

actual_payload_bpp:
  0.013999

Stage 3:
  PSNR / MS-SSIM: 21.9951 / 0.7348
  LPIPS / DISTS: 0.5758 / 0.3536

Stage 4 raw:
  PSNR / MS-SSIM: 20.5700 / 0.7002
  LPIPS / DISTS: 0.4657 / 0.2970

Stage 4 alpha0.2:
  PSNR / MS-SSIM: 21.9958 / 0.7368
  LPIPS / DISTS: 0.5560 / 0.3513
```

This alpha0.2 point was the first full552 safe internal candidate. It improved
all measured means over Stage 3 at unchanged payload, but is now superseded by
the train-cache semantic-latent + detail-context stability anchors below. It
still does not close the official CoD-Lite perceptual baseline gap, so it is not
a Stage 5 claim.

Strict full552 train-cache semantic-latent + detail-context update:

```text
train:
  20260628_stage4_cod_lite_pyramid_sem256_detailctx6_tanh075_traincache2048_5k_b8

train cache:
  20260628_stage3_gencodec512_traincache_clic_oi_2048_seed20260628

checkpoint:
  checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_tanh075_traincache2048_5k_b8.pt

eval raw:
  20260628_stage4_cod_lite_pyramid_sem256_detailctx6_tanh075_traincache2048_5k_b8_full552_eval

eval alpha0.2:
  20260628_stage4_cod_lite_pyramid_sem256_detailctx6_tanh075_traincache2048_5k_b8_full552_alpha020_eval

blend sweep:
  20260628_stage4_detailctx6_traincache2048_5k_full552_blend_sweep_patchfid256
```

Condition prediction improved substantially:

```text
base_condition_l1: 0.5371
semantic-latent Kodak24 condition_l1: 0.4711
train-cache detail-context condition_l1: 0.4112
```

Full552 condition-stat diagnosis:

```text
run:
  20260628_stage4_condition_stats_full552

actual_payload_bpp:
  0.013999

base_to_target_l1:
  0.5371

pred_to_target_l1:
  0.4112

pred L1 win rate over base:
  0.9982

base_to_target_relative_l2:
  0.8758

pred_to_target_relative_l2:
  0.6716

target / base / pred condition std:
  0.7954 / 0.8234 / 0.6932

target / base / pred spatial high-frequency ratio:
  0.2575 / 0.2776 / 0.2280
```

Interpretation:

```text
the adapter correction is real and almost always improves condition distance
the predicted condition is lower-energy and lower-frequency than target
next training should add condition cosine/stat/spectrum matching, not only L1
```

Full552 patch-FID256 sweep at unchanged actual_payload_bpp=0.013999:

```text
alpha  PSNR     MS-SSIM  LPIPS    DISTS    patch-FID
0.00   21.9951  0.7348   0.5758   0.3536   146.5134
0.20   22.0586  0.7365   0.5588   0.3547   126.1757
0.25   22.0548  0.7364   0.5508   0.3534   119.2942
0.30   22.0432  0.7360   0.5421   0.3518   112.7629
0.40   21.9964  0.7346   0.5243   0.3479   100.7237
1.00   21.1677  0.7064   0.4487   0.3127    65.1649
```

Updated interpretation:

```text
On aggregate full552, alpha0.25 and alpha0.30 supersede the earlier fixed
alpha0.2 point as the best Stage-4 decoder-side candidates. alpha0.30 improves
PSNR, MS-SSIM, LPIPS, DISTS, and patch-FID over Stage 3 without extra payload.
This is meaningful MVP progress, but it is still not an external-baseline win
over official CoD-Lite.
```

Per-split caution:

```text
alpha0.30 vs Stage 3:
  Kodak24:
    PSNR 21.6674 -> 21.6829
    MS-SSIM 0.7225 -> 0.7230
    LPIPS 0.6366 -> 0.6017
    DISTS 0.3732 -> 0.3752

  CLIC2020 test428:
    PSNR 22.3638 -> 22.4199
    MS-SSIM 0.7454 -> 0.7470
    LPIPS 0.5610 -> 0.5276
    DISTS 0.3499 -> 0.3477

  DIV2K val100:
    PSNR 20.4956 -> 20.5173
    MS-SSIM 0.6927 -> 0.6924
    LPIPS 0.6245 -> 0.5899
    DISTS 0.3648 -> 0.3639
```

So alpha0.25/0.30 are stability anchors, not final paper operating points. The
next Stage-4/5 step should remove these split-specific regressions before
claiming a robust improvement.

Condition-stat matching follow-up:

```text
train:
  20260628_stage4_cod_lite_pyramid_sem256_detailctx6_statsmatch_ft2k_b8

init:
  20260628_stage4_cod_lite_pyramid_sem256_detailctx6_tanh075_traincache2048_5k_b8.pt

loss additions:
  condition cosine weight: 0.25
  condition channel-stat weight: 0.20
  condition high-frequency-ratio weight: 0.05

checkpoint:
  checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_statsmatch_ft2k_b8.pt
```

Condition-stat result:

```text
pred_to_target_l1:
  0.4112 -> 0.4165

pred_to_target_cosine:
  0.7447 -> 0.7463

pred condition std:
  0.6932 -> 0.7282
  target: 0.7954

pred spatial high-frequency ratio:
  0.2280 -> 0.2349
  target: 0.2575
```

Interpretation:

```text
stat matching moves predicted conditions toward target energy/frequency
distribution, but slightly worsens condition L1. This confirms that condition
L1 alone is an incomplete training objective and that stat matching should be
used as part of a curriculum, not as the only promotion signal.
```

Full552 patch-FID256 sweep for the stats-match checkpoint:

```text
alpha  PSNR     MS-SSIM  LPIPS    DISTS    patch-FID
0.00   21.9951  0.7348   0.5758   0.3536   146.5134
0.25   22.0549  0.7378   0.5467   0.3510   117.3462
0.30   22.0421  0.7377   0.5372   0.3489   110.5424
0.40   21.9920  0.7368   0.5185   0.3439    98.2211
0.50   21.9103  0.7350   0.5011   0.3381    88.2460
0.75   21.5835  0.7264   0.4655   0.3214    72.1553
1.00   21.1181  0.7125   0.4428   0.3039    63.7198
```

Delta versus the previous detail-context checkpoint at the same alpha:

```text
alpha  dPSNR    dMS-SSIM  dLPIPS   dDISTS   dFID
0.25   +0.0001  +0.0014   -0.0041  -0.0024  -1.95
0.30   -0.0010  +0.0017   -0.0049  -0.0030  -2.22
0.40   -0.0044  +0.0022   -0.0059  -0.0040  -2.50
1.00   -0.0496  +0.0062   -0.0060  -0.0089  -1.45
```

Per-split alpha0.30 after stats matching:

```text
Stage 3 / alpha0.30:
  Kodak24:
    PSNR 21.6674 -> 21.6741
    MS-SSIM 0.7225 -> 0.7250
    LPIPS 0.6366 -> 0.5925
    DISTS 0.3732 -> 0.3713

  CLIC2020 test428:
    PSNR 22.3638 -> 22.4204
    MS-SSIM 0.7454 -> 0.7486
    LPIPS 0.5610 -> 0.5231
    DISTS 0.3499 -> 0.3448

  DIV2K val100:
    PSNR 20.4956 -> 20.5112
    MS-SSIM 0.6927 -> 0.6944
    LPIPS 0.6245 -> 0.5845
    DISTS 0.3648 -> 0.3610
```

This is the cleanest current full552 Stage-4 internal result: alpha0.25/0.30
now improve all measured mean metrics on Kodak24, CLIC2020 test428, and DIV2K
val100 at unchanged actual_payload_bpp. It remains a diagnostic fixed-blend
anchor, not the final method or an external-baseline win.

Oracle adaptive-alpha upper bound:

```text
invalid as a codec result:
  alpha is selected per image using reference metrics

min-LPIPS, constrained by PSNR/MS-SSIM >= Stage 3:
  full552 PSNR / MS-SSIM: 22.0349 / 0.7375
  full552 LPIPS / DISTS: 0.5455 / 0.3503

min-DISTS, constrained by PSNR/MS-SSIM >= Stage 3:
  full552 PSNR / MS-SSIM: 22.0217 / 0.7367
  full552 LPIPS / DISTS: 0.5551 / 0.3474
```

This suggests a decoder-side content gate is worth testing, but only if the
gate is computed deterministically from decoded semantic/detail features and
fixed model state. If transmitted, gate/control bits must be counted in
actual_payload_bpp.

Learned RGB output-gate diagnostic:

```text
adapter:
  checkpoints/stage4_cod_lite_adapter/20260628_stage4_cod_lite_pyramid_sem256_detailctx6_statsmatch_ft2k_b8.pt

gate checkpoint:
  checkpoints/stage4_cod_lite_gate/20260628_stage4_cod_lite_gate_statsmatch_fidelity_600_b1ga2.pt

train:
  20260628_stage4_cod_lite_gate_statsmatch_fidelity_600_b1ga2

eval:
  20260628_stage4_cod_lite_gate_statsmatch_fidelity_600_b1ga2_full552_eval

patch-FID audit:
  20260628_stage4_cod_lite_gate_statsmatch_fidelity_600_b1ga2_full552_patchfid256
```

Full552 unchanged-payload result:

```text
actual_payload_bpp:
  0.013999

Stage 3:
  PSNR / MS-SSIM: 21.9951 / 0.7348
  LPIPS / DISTS: 0.5758 / 0.3536
  patch-FID256:   146.5134

learned gate:
  alpha mean/std/min/max: 0.3071 / 0.0773 / 0.1494 / 0.4414
  PSNR / MS-SSIM:        22.0367 / 0.7384
  LPIPS / DISTS:         0.5369 / 0.3481
  patch-FID256:          109.3629
```

Comparison to the fixed alpha0.30 stats-match anchor:

```text
fixed alpha0.30:
  PSNR / MS-SSIM: 22.0421 / 0.7377
  LPIPS / DISTS:  0.5372 / 0.3489
  patch-FID256:   110.5424

learned gate:
  PSNR is slightly lower
  MS-SSIM, LPIPS, DISTS, and patch-FID are slightly better
```

Diagnostic interpretation:

```text
this is a valid no-extra-bit diagnostic and is better than hand-tuning fixed
alpha, but it still blends final RGB outputs
it is not an external-baseline win and should not be called Stage 5
do not promote this RGB output gate as the CoSER-DiC method
use it only to measure how much fidelity is being rescued by post-hoc RGB
mixing
```

Mainline condition-space gate probe:

```text
decision:
  docs/research/design_decisions/025_stage4_no_posthoc_rgb_mainline_condition_gate.md

model:
  CoSERToCoDLiteConditionGate

train:
  20260628_stage4_cod_lite_condition_gate_fidelity_probe300_b1ga2

checkpoint:
  checkpoints/stage4_cod_lite_condition_gate/20260628_stage4_cod_lite_condition_gate_fidelity_probe300_b1ga2.pt

eval:
  20260628_stage4_cod_lite_condition_gate_fidelity_probe300_b1ga2_full552_eval
```

Full552 unchanged-payload result, no RGB output blend:

```text
actual_payload_bpp:
  0.013999

Stage 3:
  PSNR / MS-SSIM: 21.9951 / 0.7348
  LPIPS / DISTS: 0.5758 / 0.3536

condition gate:
  condition_gate_mean: 0.4327
  PSNR / MS-SSIM:     21.3169 / 0.7161
  LPIPS / DISTS:      0.5255 / 0.3437
```

Mainline interpretation:

```text
condition-space gating is the right type of mechanism, but the current probe is
not faithful enough and is not promoted
the result confirms that RGB blending was masking a real CoD-Lite condition
control gap
next work should improve condition recovery and decoder conditioning, not
return to output blending
```

Detail-aware gate follow-up:

```text
adapter:
  20260628_stage4_detailaware_adapter_idstart_ft600_b2.pt

probe600 limit64:
  gate_mean: 0.9645
  PSNR / MS-SSIM: 21.6889 / 0.7397
  LPIPS / DISTS: 0.4177 / 0.3011

mean035 probe400 limit64:
  gate_mean: 0.5681
  PSNR / MS-SSIM: 21.7718 / 0.7419
  LPIPS / DISTS: 0.4684 / 0.3299

interpretation:
  raw-branch collapse can be reduced, but mean-regularized gating simply trades
  away too much perceptual gain. Neither gate is promoted.
```

CoSER feature-use ablation update:

```text
decision:
  docs/research/design_decisions/026_stage4_coser_feature_ablation_and_guarded_adapter.md

condition-stat full552:
  normal pred_to_target_l1:        0.416470
  semantic-latent zero l1:        0.429832
  detail-context zero l1:         0.417512

image eval first64:
  normal LPIPS / DISTS:           0.4969 / 0.3400
  semantic-latent zero:           0.5172 / 0.3521
  detail-context zero:            0.4986 / 0.3410

interpretation:
  decoded 256-channel semantic latent is clearly used by the adapter
  current residual_grid/detail_codes context is nearly neutral
  next mainline work should strengthen the detail-to-diffusion-control path,
  not remove the detail stream
```

Wrapper caution:

```text
CoD-Lite image decode through the current wrapper is batch-size-1 only for
image-loss training/evaluation; use gradient accumulation until this path is
made batch-safe
```

Stage 4 measured snapshot on Kodak24 512:

```text
run:
  20260628_stage4_cod_lite_adapter_lpips005_l1guard_full512_5k_kodak24_eval_save24

actual_payload_bpp:
  0.014121

Stage 3 no postprocess:
  PSNR / MS-SSIM: 21.6672 / 0.7225
  LPIPS / DISTS: 0.6366 / 0.3732
  GenCodec Kodak patch-FID: 216.7101

Stage 4 CoD-Lite adapter lpips005 l1guard:
  PSNR / MS-SSIM: 20.5132 / 0.6822
  LPIPS / DISTS: 0.4964 / 0.3151
  GenCodec Kodak patch-FID: 117.5225
```

This is a real perceptual/distribution improvement over Stage 3 at unchanged
CoSER payload, but it is not yet a large improvement over external pretrained
baselines.

Historical deterministic blend anchor:

```text
run:
  20260628_stage4_cod_lite_adapter_lpips020_alpha010_full552_eval

definition:
  stage4 = 0.9 * stage3 + 0.1 * CoD-Lite-adapter-output
  alpha is fixed decoder-side configuration, not image-specific side info

actual_payload_bpp:
  0.013999

strict 512 split:
  Kodak24 + CLIC2020 test428 + DIV2K val100
  total: 552 images

Stage 3:
  PSNR / MS-SSIM: 21.9951 / 0.7348
  LPIPS / DISTS: 0.5758 / 0.3536

Stage 4 alpha0.1:
  PSNR / MS-SSIM: 22.0147 / 0.7365
  LPIPS / DISTS: 0.5626 / 0.3529

win rates:
  PSNR:    0.7627
  MS-SSIM: 0.9601
  LPIPS:   0.9982
  DISTS:   0.5507
```

This was the first safe internal Stage 4 candidate: it improves Stage 3 on all
measured full-split means without adding transmitted bits. It is now superseded
by the train-cache semantic-latent + detail-context alpha0.25/alpha0.30 points
above. The gain is still not large enough for the Stage 5 claim, and it still
trails official CoD-Lite pretrained baselines in perceptual metrics.

Official CoD-Lite Kodak24 512 baseline curve:

```text
0.003906 bpp: PSNR 19.0304, MS-SSIM 0.5700, LPIPS 0.3390, DISTS 0.2089, FID 51.2190
0.007812 bpp: PSNR 19.7975, MS-SSIM 0.6283, LPIPS 0.2807, DISTS 0.1733, FID 44.6564
0.015625 bpp: PSNR 20.7667, MS-SSIM 0.7090, LPIPS 0.2259, DISTS 0.1402, FID 38.4785
0.031250 bpp: PSNR 21.9853, MS-SSIM 0.7811, LPIPS 0.1614, DISTS 0.1120, FID 31.8072
```

Baseline doc:

```text
docs/research/baselines/cod_lite_official_baseline_20260628.md
```

Pending full552 official baseline plan:

```text
docs/research/baselines/cod_lite_full552_official_plan_20260629.md
results/baselines/cod_lite_official/20260629_cod512_full552_plan.sh

docs/research/baselines/cod_one_step_full552_official_plan_20260629.md
results/baselines/cod_one_step_official/20260629_cod512_full552_plan.sh
```

This plan separates full552 aggregate PSNR/MS-SSIM/LPIPS/DISTS from
split-specific patch-FID: Kodak patch64, CLIC2020 test patch256, and DIV2K val
patch256. The CoD-Lite plan covers 4 rate checkpoints; the CoD one-step plan
covers the available deterministic 0.0039/0.0312/0.1250 checkpoints. They
should be run after CUDA/NVML is healthy.

Research implication:

```text
Do not claim existing-baseline superiority yet.
Use the alpha0.25/alpha0.30 detail-context blend as the current stability anchor.
The next Stage 4 design must learn a content-aware decoder-side gate or
stronger structure-preserving adapter that moves toward the official CoD-Lite
curve without losing the Stage 3 fidelity advantage.
```

## Stage 4 Mainline Update: Detail-Aware Condition Adapter

The current best no-extra-bit raw CoD-Lite condition-adapter baseline is now:

```text
run:
  20260628_stage4_detailaware_adapter_idstart_ft600_b2
  20260628_stage4_detailaware_adapter_idstart_ft3000_b2

checkpoint:
  checkpoints/stage4_cod_lite_adapter/
  20260628_stage4_detailaware_adapter_idstart_ft600_b2.pt
  20260628_stage4_detailaware_adapter_idstart_ft3000_b2.pt

design note:
  docs/research/design_decisions/027_stage4_detailaware_condition_adapter.md
  docs/research/design_decisions/028_stage4_rate_specific_cod_lite_backbone_probe.md
```

What changed:

```text
same Stage 3 semantic/detail actual payload
same frozen official CoD-Lite one-step backbone
added detail-specific condition-space residual blocks
identity-initialized new detail blocks
CoD-Lite wrapper now supports batch > 1 by sample-looping official inference
```

Full552 strict 512 result:

```text
actual_payload_bpp:
  0.013999

Stage 3:
  PSNR / MS-SSIM: 21.9951 / 0.7348
  LPIPS / DISTS: 0.5758 / 0.3536

previous stats-match raw:
  PSNR / MS-SSIM: 21.1157 / 0.7126
  LPIPS / DISTS: 0.4428 / 0.3040

detail-aware ft600 raw:
  PSNR / MS-SSIM: 21.2840 / 0.7155
  LPIPS / DISTS: 0.4358 / 0.3011

detail-aware ft3000 raw:
  PSNR / MS-SSIM: 21.3638 / 0.7182
  LPIPS / DISTS: 0.4384 / 0.3023

detail-aware bpp0.0312 transfer raw:
  PSNR / MS-SSIM: 21.4438 / 0.7216
  LPIPS / DISTS: 0.4402 / 0.2917

detail-aware fast8192 ft1500 raw:
  PSNR / MS-SSIM: 21.3957 / 0.7186
  LPIPS / DISTS: 0.4342 / 0.3022
  condition_l1: 0.4095
```

Delta vs previous stats-match raw:

```text
ft600:
PSNR:    +0.1683 dB
MS-SSIM: +0.0029
LPIPS:   -0.0069
DISTS:   -0.0029

ft3000:
PSNR:    +0.2481 dB
MS-SSIM: +0.0056
LPIPS:   -0.0044
DISTS:   -0.0017
```

GenCodec-style patch-FID/KID, torch-fidelity backend:

```text
Kodak24:
  patch_size / fid_patch_num: 64 / 2
  patches: 2712
  FID / KID: 101.5062 / 0.037533

CLIC2020 test428:
  patch_size / fid_patch_num: 256 / 2
  patches: 2140
  FID / KID: 67.3085 / 0.011038

DIV2K val100:
  patch_size / fid_patch_num: 256 / 2
  patches: 500
  FID / KID: 169.1644 / 0.021140
```

Rate-specific bpp0.0312 transfer patch-FID/KID:

```text
Kodak24:
  patch_size / fid_patch_num: 64 / 2
  patches: 2712
  FID / KID: 104.7621 / 0.036518

CLIC2020 test428:
  patch_size / fid_patch_num: 256 / 2
  patches: 2140
  FID / KID: 87.1151 / 0.025608

DIV2K val100:
  patch_size / fid_patch_num: 256 / 2
  patches: 500
  FID / KID: 182.9174 / 0.038770
```

Fast 8192-image train-cache continuation patch-FID/KID:

```text
Kodak24:
  patch_size / fid_patch_num: 64 / 2
  patches: 2712
  FID / KID: 102.8851 / 0.038774

CLIC2020 test428:
  patch_size / fid_patch_num: 256 / 2
  patches: 2140
  FID / KID: 66.7065 / 0.010536

DIV2K val100:
  patch_size / fid_patch_num: 256 / 2
  patches: 500
  FID / KID: 167.7346 / 0.019636
```

Current judgment:

```text
This is a real mainline Stage 4 improvement because it changes decoded CoSER
feature use in the condition adapter rather than blending final RGB outputs.
ft600 is the current perceptual-leaning raw anchor.
ft3000 is the bpp0.0156 continuation / fidelity-recovery check.
fast8192 ft1500 is the current LPIPS/condition-recovery raw anchor, with
better CLIC/DIV2K patch-FID/KID but weaker Kodak patch-FID/KID than ft600.
bpp0.0312 transfer is the current DISTS/fidelity-leaning backbone anchor, but it
regresses LPIPS and patch-FID/KID relative to ft600.
The ft600 -> ft3000 and bpp0.0156 -> bpp0.0312 tradeoffs show that longer
training or stronger pretrained prior alone is not enough.
It is not Stage 5 and still does not beat official CoD-Lite baselines.
```

## Stage 4 Fast8192 Feature-Use Ablation

Checkpoint:

```text
20260629_stage4_detailaware_ft600_fast8192_ft1500_b4.pt
```

Full552 ablation:

```text
normal:
  PSNR / MS-SSIM: 21.3957 / 0.7186
  LPIPS / DISTS: 0.4342 / 0.3022
  condition_l1:   0.4095

semantic latent zero:
  PSNR / MS-SSIM: 20.3170 / 0.6961
  LPIPS / DISTS: 0.4624 / 0.3396
  condition_l1:   0.4269

semantic latent shuffle:
  PSNR / MS-SSIM: 20.1303 / 0.6990
  LPIPS / DISTS: 0.4607 / 0.3173
  condition_l1:   0.4339

detail context zero:
  PSNR / MS-SSIM: 21.4080 / 0.7182
  LPIPS / DISTS: 0.4394 / 0.3061
  condition_l1:   0.4105

detail context shuffle:
  PSNR / MS-SSIM: 21.3836 / 0.7179
  LPIPS / DISTS: 0.4377 / 0.3045
  condition_l1:   0.4107
```

Judgment:

```text
semantic latent is being used strongly and image-specifically
detail context is useful but still weak as a diffusion-control signal
next mainline work should strengthen detail-aware condition recovery
```

## Stage 4 Semantic-Dropout Detail-Use Probe

Semantic-latent dropout was added as a training-only regularizer:

```text
script:
  scripts/train_stage4_cod_lite_adapter.py

args:
  --semantic-latent-dropout-prob
  --detail-context-dropout-prob
```

Main probe:

```text
run:
  20260629_stage4_fast8192_ft1500_semdrop020_ft800_b4

checkpoint:
  checkpoints/stage4_cod_lite_adapter/
  20260629_stage4_fast8192_ft1500_semdrop020_ft800_b4.pt

init:
  20260629_stage4_detailaware_ft600_fast8192_ft1500_b4.pt

semantic_latent_dropout_prob:
  0.20
```

Full552 result at unchanged actual_payload_bpp=0.013999:

```text
fast8192 ft1500:
  PSNR / MS-SSIM: 21.3957 / 0.7186
  LPIPS / DISTS: 0.4342 / 0.3022
  condition_l1:   0.4095

semantic dropout ft800:
  PSNR / MS-SSIM: 21.2818 / 0.7186
  LPIPS / DISTS: 0.4325 / 0.2982
  condition_l1:   0.4092
```

Feature-use ablation after semantic dropout:

```text
normal:
  PSNR / MS-SSIM: 21.2818 / 0.7186
  LPIPS / DISTS: 0.4325 / 0.2982
  condition_l1:   0.4092

semantic latent zero:
  PSNR / MS-SSIM: 21.1284 / 0.7102
  LPIPS / DISTS: 0.4359 / 0.3084
  condition_l1:   0.4118

semantic latent shuffle:
  PSNR / MS-SSIM: 20.8946 / 0.7091
  LPIPS / DISTS: 0.4418 / 0.3036
  condition_l1:   0.4192

detail context zero:
  PSNR / MS-SSIM: 21.3055 / 0.7184
  LPIPS / DISTS: 0.4372 / 0.3017
  condition_l1:   0.4100

detail context shuffle:
  PSNR / MS-SSIM: 21.2830 / 0.7181
  LPIPS / DISTS: 0.4356 / 0.3004
  condition_l1:   0.4101
```

Judgment:

```text
semantic dropout improves LPIPS/DISTS slightly but loses PSNR
semantic latent remains the dominant CoSER-to-CoD-Lite condition signal
detail context is still under-used
do not promote this checkpoint as a Stage 5 result or external-baseline win
keep dropout infrastructure, but move next to a stronger detail-control branch
```

## Stage 4 Detail-Control Branch Probe

The pyramid adapter now has an optional direct detail condition branch:

```text
script:
  scripts/train_stage4_cod_lite_adapter.py

arg:
  --detail-control-branch

implementation:
  src/coserdic/models/gencodec_backbone.py
```

It uses the same decoded detail payload features:

```text
residual_grid_hat
normalized detail_codes
```

No extra image-specific payload is transmitted.

Main probe:

```text
run:
  20260629_stage4_fast8192_detailcontrol_ft800_b4

checkpoint:
  checkpoints/stage4_cod_lite_adapter/
  20260629_stage4_fast8192_detailcontrol_ft800_b4.pt

init:
  20260629_stage4_detailaware_ft600_fast8192_ft1500_b4.pt
```

Full552 result at unchanged actual_payload_bpp=0.013999:

```text
fast8192 ft1500:
  PSNR / MS-SSIM: 21.3957 / 0.7186
  LPIPS / DISTS: 0.4342 / 0.3022
  condition_l1:   0.4095

semantic dropout ft800:
  PSNR / MS-SSIM: 21.2818 / 0.7186
  LPIPS / DISTS: 0.4325 / 0.2982
  condition_l1:   0.4092

detail-control ft800:
  PSNR / MS-SSIM: 21.3444 / 0.7160
  LPIPS / DISTS: 0.4325 / 0.3013
  condition_l1:   0.4071
```

Detail ablation for the detail-control checkpoint:

```text
normal:
  PSNR / MS-SSIM: 21.3444 / 0.7160
  LPIPS / DISTS: 0.4325 / 0.3013
  condition_l1:   0.4071

detail context zero:
  PSNR / MS-SSIM: 21.3575 / 0.7156
  LPIPS / DISTS: 0.4377 / 0.3053
  condition_l1:   0.4081

detail context shuffle:
  PSNR / MS-SSIM: 21.3339 / 0.7153
  LPIPS / DISTS: 0.4360 / 0.3039
  condition_l1:   0.4083
```

Judgment:

```text
direct detail condition injection is valid infrastructure
condition_l1 improves
full552 image metrics do not dominate previous anchors
detail zero/shuffle damage is nearly unchanged from previous adapters
do not promote this checkpoint as Stage 5 or external-baseline progress
next mainline step should strengthen detail-specific condition supervision or
condition-space detail modulation
```

Semantic-dropout follow-up on the detail-control branch:

```text
run:
  20260629_stage4_fast8192_detailcontrol_semdrop020_ft800_b8

batch:
  8

semantic_latent_dropout_prob:
  0.20
```

Full552 result:

```text
detail-control + semdrop b8:
  PSNR / MS-SSIM: 21.2325 / 0.7154
  LPIPS / DISTS: 0.4315 / 0.2988
  condition_l1:   0.4069
```

Detail ablation:

```text
normal:
  LPIPS / DISTS / condition_l1: 0.4315 / 0.2988 / 0.4069

detail context zero:
  LPIPS / DISTS / condition_l1: 0.4361 / 0.3023 / 0.4079

detail context shuffle:
  LPIPS / DISTS / condition_l1: 0.4345 / 0.3009 / 0.4080
```

Judgment:

```text
best raw LPIPS and condition_l1 among the fast8192 anchors
does not beat semantic dropout alone on DISTS
further hurts PSNR/MS-SSIM
detail ablation sensitivity remains mild
not promoted
```

## Stage 4 DISTS-Loss Probe Status

The adapter trainer now supports a DISTS image loss through:

```text
--dists-weight
```

First guarded continuation:

```text
run:
  20260629_stage4_detailaware_ft600_distsguard_ft1000_b2

init:
  20260628_stage4_detailaware_adapter_idstart_ft600_b2.pt

limit64 eval:
  actual_payload_bpp = 0.013654
  PSNR / MS-SSIM = 21.8067 / 0.7420
  LPIPS / DISTS = 0.4195 / 0.3035
```

Judgment:

```text
not promoted

The DISTS hook is useful infrastructure, but this probe is dominated by ft600
on LPIPS/DISTS and by bpp0.0312 transfer on DISTS/fidelity for the same limit64
subset. It does not change the conclusion that the bottleneck is condition
recovery and diffusion control quality, not merely another image-space loss.
```

Next mainline move:

```text
scale from the 2048-image train-cache to larger clean CLIC/OpenImages crops
before drawing ceiling conclusions about the condition adapter
```

## Stage 5 Definition

Stage 5 starts only after a Stage 4 diffusion path works. It is not simply
"train longer".

Stage 5 means joint Core MVP finetuning:

```text
semantic encoder
detail residual encoder
entropy priors
auxiliary decoder
diffusion adapter
optional very-low-lr diffusion backbone
```

Required ablations before a Stage 5 claim:

```text
semantic only
semantic + detail, no diffusion
diffusion without detail
diffusion without semantic conditioning
CoD-Lite vs CoD backbone
actual_payload_bpp and estimated_bpp reported separately
```

## 2026-06-29 Stage 5 Counted-Control Status

Current Stage 5 work is focused on counted condition-control streams for the
CoD-Lite adapter path, not fixed RGB output edits.

Implemented CPU-side infrastructure:

```text
affine condition control
affine + DCT residual control
affine + grid residual control
post-affine basis residual control
hybrid per-image control selection
image-RDO and guarded image-RDO selection
component p95/p99 basis quantization
component codebook basis quantization
component codebook Huffman basis coding
vector codebook basis control
compact fixed-bit sparse top-k basis payload packing
anchor-aware limit64 selection and full552 promotion planning
```

All per-image control streams are counted in `actual_payload_bpp`. Fixed bases,
Huffman priors, component ranges, and component codebooks are decoder-side
state fitted from non-eval train-cache data.

GPU remains unavailable in the current container state:

```text
nvidia-smi: Failed to initialize NVML: Unknown Error
```

Do not launch Stage 5 training/evaluation until the container is restarted and
`scripts/check_gpu_ready.py --min-devices 1` passes.

## Main References

```text
Stage 1-3 active anchor:
  docs/research/design_decisions/019_stage1_rateprior_b16_reconnection.md

Refiner archive:
  docs/research/design_decisions/020_stage4_decoder_side_refiner.md

Active Stage 4 policy:
  docs/research/design_decisions/021_stage4_cod_codlite_parallel_backbone_policy.md

Stage 4 adapter bootstrap:
  docs/research/design_decisions/022_stage4_cod_lite_adapter_bootstrap.md

Stage 4 semantic-latent adapter:
  docs/research/design_decisions/023_stage4_semantic_latent_cod_lite_adapter.md

Stage 4/5 mainline direction:
  docs/research/design_decisions/024_stage4_stage5_mainline_research_direction.md

Stage 4 detail-aware adapter:
  docs/research/design_decisions/027_stage4_detailaware_condition_adapter.md

Stage 4 rate-specific CoD-Lite probe:
  docs/research/design_decisions/028_stage4_rate_specific_cod_lite_backbone_probe.md

Stage 4 DISTS-loss probe and train-cache scale:
  docs/research/design_decisions/029_stage4_dists_loss_probe_and_traincache_scale.md

Stage 4 fast train-cache and 8192 adapter scale:
  docs/research/design_decisions/030_stage4_fast_train_cache_and_8192_adapter_scale.md

Stage 4 fast8192 CoSER feature ablation:
  docs/research/design_decisions/031_stage4_fast8192_coser_feature_ablation.md

Stage 4 semantic dropout and detail-use probe:
  docs/research/design_decisions/032_stage4_semantic_dropout_detail_use_probe.md

Stage 5 counted-control master path:
  docs/research/design_decisions/048_stage5_counted_control_master_screen_selection.md

Stage 5 component codebook basis quantization:
  docs/research/design_decisions/055_stage5_component_codebook_basis_quantization.md

Stage 5 component codebook Huffman control:
  docs/research/design_decisions/056_stage5_component_codebook_huffman_control.md

Stage 5 vector codebook basis control:
  docs/research/design_decisions/057_stage5_vector_codebook_basis_control.md

Stage 5 compact sparse top-k basis payload:
  docs/research/design_decisions/058_stage5_compact_sparse_topk_basis_payload.md

Stage 4 perceptual adapter and CoSER feature ablation:
  docs/research/design_decisions/060_stage4_perceptual_adapter_and_coser_feature_ablation.md

Pretrained inventory:
  docs/research/baselines/pretrained_asset_inventory_20260628.md

Bpp policy:
  docs/research/design_decisions/010_bpp_reporting_policy.md
```

## 2026-06-29 Active Update: Mainline Stage 4/5 Anchor

The earlier note that GPU was unavailable is now stale for this session:

```text
GPU:
  NVIDIA GeForce RTX 4070 Ti SUPER
  visible and usable during the 2026-06-29 Stage 4/5 runs
```

Current best no-extra-bit Stage 4 CoD-Lite adapter:

```text
checkpoint:
  checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailfilm_dists_perceptual_ft1200_b8.pt

full552 eval:
  results/stage4_cod_lite_adapter_eval/20260629_stage4_detailfilm_dists_perceptual_ft1200_b8_full552_eval/summary.json

actual_payload_bpp:
  0.013999

metrics:
  PSNR     21.2237
  MS-SSIM   0.7162
  LPIPS     0.4159
  DISTS     0.2830
```

Current best internal counted-control Stage 5 candidate:

```text
run:
  20260629_stage5_perceptual_g32basis_full552_affinebasis_ag16_as1_bg32_bs8_k32_b4_topkc256_gr1_br0.25_r0.710860_sc1_huffman_mu_law_mu16_topk_c256_k32_p99_b4_mulaw16

actual_payload_bpp:
  0.015485

control_payload_bpp:
  0.001486

metrics:
  PSNR     21.1592
  MS-SSIM   0.7148
  LPIPS     0.3918
  DISTS     0.2707
```

This counted-control point improves the previous best top-k32 counted-control
anchor at essentially the same bpp, but it is still not an external-baseline
win against official CoD-Lite. Continue to describe it as internal Stage 5
progress unless full official-baseline curves say otherwise.

Feature ablations on limit64 show:

```text
delta_zero:
  LPIPS +0.1602, DISTS +0.0839, condition_l1 +0.1170

semantic_shuffle:
  PSNR -0.1039, MS-SSIM -0.0062, LPIPS +0.0055, DISTS +0.0009

detail_shuffle:
  LPIPS +0.0018, DISTS +0.0021
```

Interpretation:

```text
adapter delta is essential.
semantic latent carries useful image-specific structure.
detail context helps perceptual quality but currently trades off PSNR.
```

Next mainline target:

```text
Improve the diffusion-control detail pathway using the same transmitted detail
payload, then refit counted-control only after the no-extra adapter improves.
Do not return to fixed RGB output blending as the method.
```

## 2026-06-29 Active Update: Detail-Contrast Stage 4 and Sparse Stage 5 Anchor

The previous `detailfilm_dists_perceptual_ft1200_b8` anchor has been superseded
by a detail-contrast / high-frequency residual continuation:

```text
checkpoint:
  checkpoints/stage4_cod_lite_adapter/20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8.pt

full552 eval:
  results/stage4_cod_lite_adapter_eval/20260629_stage4_detailfilm_detailcontrast_hf_ft800_b8_full552_eval/summary.json

actual_payload_bpp:
  0.013999

metrics:
  PSNR     21.2351
  MS-SSIM   0.7166
  LPIPS     0.4139
  DISTS     0.2821
  condition_l1 0.4108
```

Delta versus the previous no-extra perceptual anchor:

```text
PSNR     +0.0114
MS-SSIM  +0.0004
LPIPS    -0.0020
DISTS    -0.0009
condition_l1 -0.0017
```

This is now the current best no-extra-bit Stage 4 CoD-Lite adapter.

The matching counted-control basis is:

```text
outputs/stage5_control_basis/20260629_detailcontrast_hf_postaffine_basis_g32s8_k256_train8192_b12/control_basis.pt

coefficient p99:
  0.702655
```

Current balanced internal Stage 5 counted-control anchor:

```text
run:
  20260629_stage5_detailcontrast_hf_g32basis_full552_affinebasis_ag16_as1_bg32_bs8_k32_b4_topkc256_gr1_br0.25_r0.702655_sc1_huffman_mu_law_mu16_topk_c256_k32_p99_b4_mulaw16

actual_payload_bpp:
  0.015486

metrics:
  PSNR     21.1617
  MS-SSIM   0.7149
  LPIPS     0.3891
  DISTS     0.2691
  condition_l1 0.4018
```

Delta versus the previous counted-control anchor:

```text
bpp      +0.000001
PSNR     +0.0025
MS-SSIM  +0.0001
LPIPS    -0.0028
DISTS    -0.0015
condition_l1 -0.0018
```

Current perceptual-oriented operating point:

```text
run:
  20260629_stage5_detailcontrast_hf_g32basis_full552_affinebasis_ag16_as1_bg32_bs8_k32_b4_topkc256_gr1_br0.25_r0.702655_sc125_huffman_mu_law_mu16_topk_c256_k32_p99_b4_mulaw16

actual_payload_bpp:
  0.015488

metrics:
  PSNR     21.1319
  MS-SSIM   0.7140
  LPIPS     0.3844
  DISTS     0.2663
  condition_l1 0.4027
```

GenCodec-style patch-FID/KID for the `control_scale=1.25` saved full552 run:

```text
saved reconstructions:
  results/stage4_cod_lite_adapter_eval/20260629_stage5_detailcontrast_hf_g32basis_full552_sc125_save_eval/

protocol:
  Kodak: patch64, fid_patch_num=2
  CLIC2020 test / DIV2K val: patch256, fid_patch_num=2
  backend: torch-fidelity
```

| dataset | Stage3 FID | Stage5 sc1.25 FID | delta | Stage3 KID | Stage5 sc1.25 KID |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kodak24 | 216.3567 | 79.2004 | -137.1562 | 0.150822 | 0.020942 |
| CLIC2020 test428 | 152.0035 | 57.4915 | -94.5120 | 0.079694 | 0.007016 |
| DIV2K val100 | 234.6295 | 152.6266 | -82.0029 | 0.092627 | 0.013602 |

Interpretation:

```text
The counted condition-control stream substantially improves Stage3 distribution
metrics on every strict full552 split, but it still does not close the official
CoD-Lite perceptual/FID gap.

The fixed control_scale sweep shows a real fidelity/perception tradeoff:
  scale=1.0 is better balanced and better condition_l1.
  scale=1.25 gives better LPIPS/DISTS/FID/KID with more fidelity loss.

The next mainline move is therefore a learned deterministic content gate for
condition residual strength, not more fixed-scale or RGB-output tweaking.
```

Detailed record:

```text
docs/research/design_decisions/061_stage4_detailcontrast_and_stage5_sparse_control.md
```

## 2026-06-29 Stage 5 control-aware adapter training probe

Implemented training-time counted control support:

```text
script:
  scripts/train_stage4_cod_lite_adapter.py

new option:
  --train-counted-control-mode condition_residual_affine_basis
```

Purpose:

```text
Train the CoSER-to-CoD-Lite adapter with the same entropy-coded affine+basis
condition-control path used at evaluation time, so the adapter is optimized for
an actual transmitted control operating point rather than only a continuous
teacher condition.
```

Run:

```text
20260629_stage5_controlaware_adapter_lora0156_all_k16_ft400_b4ga2

checkpoint:
  checkpoints/stage4_cod_lite_adapter/20260629_stage5_controlaware_adapter_lora0156_all_k16_ft400_b4ga2.pt

training summary:
  results/stage4_cod_lite_adapter/20260629_stage5_controlaware_adapter_lora0156_all_k16_ft400_b4ga2/summary.json
```

Training diagnostic:

```text
pre_control_condition_l1_mean:
  0.3934

condition_l1_mean after counted control:
  0.3873

control_condition_l1_delta_mean:
  -0.0061

control_payload_bpp_512_mean:
  0.000990
```

Full552 comparison:

| run | bpp | PSNR | MS-SSIM | LPIPS | DISTS | condition L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA-all no-extra | 0.013999 | 21.2926 | 0.7149 | 0.3867 | 0.2686 | 0.4113 |
| LoRA-all + k16 control | 0.014980 | 21.1959 | 0.7116 | 0.3664 | 0.2574 | 0.4030 |
| control-aware FT400, no control | 0.013999 | 21.3594 | 0.7166 | 0.3888 | 0.2716 | 0.4106 |
| control-aware FT400 + k16 control | 0.014978 | 21.2492 | 0.7130 | 0.3655 | 0.2589 | 0.4017 |
| pre-control guarded FT300 + k16 control | 0.014978 | 21.2452 | 0.7121 | 0.3658 | 0.2591 | 0.4013 |

GenCodec-style patch-FID/KID for control-aware FT400 + k16 control:

| split | FID | KID |
| --- | ---: | ---: |
| Kodak24 | 80.9884 | 0.020049 |
| CLIC2020 test428 | 62.7218 | 0.010622 |
| DIV2K val100 | 161.8326 | 0.021273 |

Decision:

```text
Do not promote this checkpoint.

It is useful infrastructure because training-time counted control now works,
but this exact loss recipe is not a good long-run seed:
  - raw adapter moved toward fidelity but worsened LPIPS/DISTS
  - counted control recovers LPIPS but DISTS is worse than the previous k16 point
  - pre-control condition L1 guard slightly improves condition L1 but still
    does not recover DISTS
  - CLIC/DIV2K patch-FID/KID are worse than the stronger existing
    detailtarget long balanced/postgate anchors
```

Next mainline implication:

```text
The adapter should learn a residual that is complementary to the fixed k16
control basis, not rely on a control correction that chases a moving target.

Next experiments should focus on:
  - complementary residual/control-aware loss
  - stronger DISTS/FID-aware but fidelity-guarded condition training
  - deterministic condition/control-space content gate
  - Stage 3 diffusion-friendly detail-control heads
```

Detailed record:

```text
docs/research/design_decisions/069_stage5_controlaware_adapter_training_probe_20260629.md
```
