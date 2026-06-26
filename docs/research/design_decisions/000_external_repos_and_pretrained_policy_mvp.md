# CoSER-DiC MVP-v0: External Repositories and Pretrained Usage Policy

Date: 2026-06-26  
Status: Required design-decision memo before starting research implementation  
Target project: `https://github.com/yutendo0714/CoSER-DiC`

---

## 0. この文書の目的

この文書は、CoSER-DiC研究を進める前に、外部GitHubリポジトリ、pretrained weights、MVP実装範囲を明確に固定するための設計決定メモである。

元のCoSER-DiC提案パッケージには、full versionに向けた複数の候補や拡張案が含まれている。研究開始時点では、それらをすべて同時に実装しない。まずは **MVP-v0** に絞って、CoSER-DiCの中核仮説を最小構成で検証する。

この文書で固定する方針は以下である。

```text
CoSER-DiCは新規codecとして実装する。
外部repoはbaseline、pretrained initialization、設計参照、sanity checkに使う。
外部repoをそのままCoSER-DiC本体にしない。
MVP-v0では、semantic VQ / detail residual / entropy model / auxiliary decoderはCoSER-DiC側で新規学習する。
Diffusion decoder backboneのみ、CoD-Lite pretrainedを第一候補として初期化に使う。
```

---

## 1. まず確認すべき結論

はい、方針は **MVP-v0に絞る**。

元パッケージのfull versionには以下のような発展要素が含まれている。

- VLM-DPO
- OCR loss
- face identity loss
- adaptive quantization
- variable-rate control
- human 2AFC study
- Mamba / full autoregressive entropy model
- complex semantic-critical evaluation

しかし、これらは最初から実装しない。MVP-v0では、以下の中核だけを実装・検証する。

```text
1. RD-aware Semantic VQ token stream
2. Detail residual latent stream
3. Semantic-conditioned residual entropy model
4. Auxiliary reconstruction
5. Compression-oriented one-step diffusion reconstruction
6. Actual entropy-coded bitstream and actual bpp reporting
```

この6つを外したものは、CoSER-DiC MVPとは呼ばない。

---

## 2. CoSER-DiCの中核主張

CoSER-DiCの主張は、単に「既存のVQ codecとdiffusion codecを組み合わせる」ことではない。

悪い解釈:

```text
CoSER-DiC = RDVQ + CoD-Lite
CoSER-DiC = StableCodecの改造版
CoSER-DiC = 既存diffusion codecにVQ branchを足したもの
```

正しい解釈:

```text
Extreme-low-bitrate generative image compression benefits from separating semantic structure,
detail-critical residuals, and stochastic texture generation under entropy-constrained coding.
```

日本語では以下である。

```text
極低bpp生成圧縮では、semantic structure、detail-critical residual、stochastic textureを
単一latentに混在させるのではなく、役割分担してentropy制約下で符号化・生成する方が、
faithfulnessとrealismの両方を改善できる。
```

したがって、実装では次の役割分担を必ず維持する。

| Component | 役割 |
|---|---|
| Semantic VQ token stream | 大域構造、scene layout、object layout、semantic structureを少ないbitで保持する |
| Detail residual latent stream | 顔、文字、小物、境界、identityなど、semantic tokenだけでは保持できないcritical detailsを補う |
| Semantic-conditioned residual entropy model | residual branchがsemantic tokenで説明済みの情報を重複して送らないようにする |
| Auxiliary reconstruction | diffusion decoderへのfidelity anchor。生成に丸投げしないための中間復元 |
| Compression-oriented diffusion decoder | bitstreamで送らなかった自然なtexture / high-frequency detailを生成する |
| Actual bitstream / actual bpp | 論文用評価ではestimated bppではなく、実際のentropy-coded bitstreamを使う |

---

## 3. 現在のプロジェクト構成に対する前提

研究者が `https://github.com/yutendo0714/CoSER-DiC` に作成したプロジェクト構成は、おおむね正しい前提で進める。

期待する構成は以下である。

```text
src/coserdic/                         # CoSER-DiC本体実装
docs/specs/coserdic_package_v1/       # 元提案パッケージ・凍結仕様
external/repos/                       # cloneした外部GitHub repo。git管理しない
external/pretrained/                  # 外部pretrained weights。git管理しない
checkpoints/                          # CoSER-DiCの学習済みcheckpoint。git管理しない
bitstreams/                           # actual bitstream examples。git管理しない
results/                              # metrics, tables, reconstructed images。git管理しない
configs/                              # training/evaluation/baseline config
scripts/                              # train/eval/compress/decompress scripts
docs/research/                        # 研究ログ、design decision、baseline registry
```

この文書は、以下のような場所に追加することを推奨する。

```text
docs/research/design_decisions/000_external_repos_and_pretrained_policy_mvp.md
```

---

## 4. MVP-v0で実装するもの

MVP-v0では、以下だけを実装する。

```text
1. SemanticEncoder
2. DifferentiableVQ
3. SemanticCategoricalEntropyModel
4. SemanticAuxiliaryDecoder
5. DetailResidualEncoder
6. DetailGaussianEntropyModel
7. SemanticConditionedResidualPrior
8. JointAuxiliaryDecoder
9. DiffusionDecoder initialized from CoD_Lite_pretrain.pt if feasible
10. CoSER-specific conditioning adapter / FiLM / control branch
11. compress()
12. decompress()
13. actual bitstream round-trip tests
14. estimated bpp and actual bpp reporting
```

MVP-v0の初期rate pointsは以下に絞る。

```text
0.01 bpp
0.03 bpp
0.05 bpp
```

理由:

```text
0.003 / 0.005 bppは非常に難度が高い。
まず0.01〜0.05 bppでsemantic/detail/diffusion分離の有効性を確認する。
MVPが安定した後に0.003 / 0.005 bppへ拡張する。
```

---

## 5. MVP-v0では実装しないもの

以下はfull versionまたはpost-MVP extensionであり、MVP-v0では実装しない。

```text
1. VLM-DPO
2. OCR loss
3. face identity loss
4. transmitted segmentation maps
5. transmitted captions
6. transmitted OCR masks
7. transmitted importance maps
8. complex adaptive quantization
9. full autoregressive semantic token prior
10. Mamba / SSM entropy model
11. multi-step diffusion with many denoising steps
12. human 2AFC study
13. advanced variable-rate control
14. large-scale semantic-critical benchmark construction
```

これらを最初から入れると、以下の問題が起きる。

```text
実装が複雑化する。
学習が不安定になる。
どのmoduleが効いたか分からなくなる。
論文の主張が散らばる。
CoSER-DiCの中核が「全部入りシステム」に見える。
```

---

## 6. 外部GitHub repoの基本方針

CoSER-DiC本体は、既存repoのforkではなく、新規PyTorch codebaseとして `src/coserdic` に実装する。

外部repoは次の用途に限定する。

```text
1. baseline reproduction
2. pretrained diffusion decoder initialization
3. implementation reference
4. sanity check
5. optional ablation
```

外部repoをそのままCoSER-DiCの本体にしてはいけない。

---

## 7. Primary external repositories

### 7.1 microsoft/GenCodec / CoD_Lite

Repository:

```text
https://github.com/microsoft/GenCodec/tree/main/CoD_Lite
```

Role:

```text
Primary diffusion decoder initialization candidate.
Primary one-step diffusion baseline.
Real-time / lightweight diffusion design reference.
```

予定pretrained:

```text
zhaoyangjia/CoD_Lite:
  CoD_Lite_pretrain.pt
  CoD_Lite_bpp_0_0039.pt
  CoD_Lite_bpp_0_0078.pt
  CoD_Lite_bpp_0_0156.pt
  CoD_Lite_bpp_0_0312.pt
```

Usage policy:

```text
CoD_Lite_pretrain.pt:
  CoSER-DiCのdiffusion reconstruction moduleの第一初期化候補として使う。

CoD_Lite_bpp_* checkpoints:
  baselineおよびsanity checkとして使う。
  CoSER-DiC本体の最終モデルとは扱わない。

CoD-Lite architecture:
  decoder backbone / design referenceとして使う。
  CoSER-DiC側で、x_aux, s_hat, d_hat, rate conditionを注入するadapterを新規実装する。
```

重要:

```text
CoD-Lite decoderは、もともとCoSER-DiCのsemantic/detail dual representationを受け取る設計ではない。
そのため、直接互換性を仮定しない。
CoSER-specific adapter, FiLM, control branchを必ず実装する。
```

Expected paths:

```text
external/repos/GenCodec/
external/pretrained/CoD_Lite/
src/coserdic/models/diffusion_decoder.py
src/coserdic/models/conditioning_adapter.py
```

---

### 7.2 CVL-UESTC/RDVQ

Repository:

```text
https://github.com/CVL-UESTC/RDVQ
```

Role:

```text
Primary VQ-based generative compression baseline.
Reference for differentiable / RD-aware VQ.
Reference for estimated vs real-bitstream evaluation.
Optional semantic tokenizer initialization ablation after MVP is stable.
```

予定pretrained:

```text
RDVQ official Hugging Face checkpoints
```

Usage policy:

```text
MVP-v0:
  CoSER-DiC semantic VQをRDVQ pretrainedから初期化しない。
  CoSER-DiC semantic VQはscratchで学習する。

After MVP-v0:
  tensor shapeと設計が合う場合のみ、optional ablationとしてRDVQ tokenizer initializationを試す。

RDVQ checkpoints:
  baseline比較に使う。
  estimated bpp / actual bpp評価pipelineのsanity checkに使う。
```

Reason:

```text
RDVQはCoSER-DiCのsemantic VQ branchに非常に近い。
最初からRDVQ pretrainedを強く使うと、CoSER-DiCが「RDVQ + diffusion」に見える危険がある。
新規性を守るため、defaultではsemantic VQをCoSER-DiCとしてscratch学習する。
```

Expected paths:

```text
external/repos/RDVQ/
external/pretrained/RDVQ/
configs/baselines/rdvq.yaml
docs/research/baselines/rdvq_reproduction.md
```

---

### 7.3 InterDigitalInc/CompressAI

Repository:

```text
https://github.com/interdigitalinc/compressai
```

Role:

```text
RD-oriented learned image compression baselines.
Entropy bottleneck / Gaussian conditional implementation reference.
Evaluation pipeline reference.
```

予定pretrained models:

```text
bmshj2018-factorized
bmshj2018-hyperprior
mbt2018-mean
mbt2018
cheng2020-anchor
cheng2020-attn
```

Usage policy:

```text
CompressAIはbaselineと実装参考に使う。
CoSER-DiC本体をCompressAI model subclassとして実装することは原則避ける。

CoSER-DiC本体では、以下のstream decompositionを明示的に維持する。
  - semantic token stream
  - semantic hyper stream
  - detail latent stream
  - detail hyper stream
  - header / optional control stream
```

Expected paths:

```text
external/repos/compressai/
configs/baselines/compressai_cheng2020.yaml
configs/baselines/compressai_hyperprior.yaml
scripts/run_compressai_baselines.py
```

---

## 8. Secondary / fallback repositories

### 8.1 LuizScarlet/StableCodec

Repository:

```text
https://github.com/LuizScarlet/StableCodec
```

Role:

```text
Strong one-step diffusion extreme image compression baseline.
Fallback design reference if CoD-Lite integration fails.
Reference for auxiliary reconstruction / dual-branch coding.
```

予定pretrained:

```text
SD-Turbo
stablecodec_base.pkl
stablecodec_ft2.pkl
stablecodec_ft3.pkl
stablecodec_ft4.pkl
stablecodec_ft6.pkl
stablecodec_ft8.pkl
stablecodec_ft12.pkl
stablecodec_ft16.pkl
stablecodec_ft24.pkl
stablecodec_ft32.pkl
elic_official.pth
```

Usage policy:

```text
StableCodecはbaselineとして使う。
CoD-Lite integrationが詰まった場合のみ、fallback decoder referenceとして検討する。
MVP-v0のdefault backboneにはしない。
```

Reason:

```text
StableCodecはそれ自体が完成されたone-step diffusion codecである。
依存しすぎると、CoSER-DiCがStableCodec派生に見える危険がある。
```

---

### 8.2 LuizScarlet/AEIC

Repository:

```text
https://github.com/LuizScarlet/AEIC
```

Role:

```text
Speed / shallow encoder / asymmetric extreme image compression baseline.
Reference for fast encoding design.
```

Usage policy:

```text
baselineとして使う。
CoSER-DiCのdefault initializationには使わない。
```

---

### 8.3 onedc-codec/onedc

Repository:

```text
https://github.com/onedc-codec/onedc
```

Role:

```text
One-step diffusion codec baseline.
Reference for semantic consistency and one-step diffusion design.
```

Usage policy:

```text
baselineおよび設計参照として使う。
license compatibilityを確認せずにCoSER-DiC coreへ直接統合しない。
```

---

### 8.4 jzyustc/GLC

Repository:

```text
https://github.com/jzyustc/GLC
```

Role:

```text
VQ / generative latent coding baseline.
Reference for generative latent compression.
```

Usage policy:

```text
baselineとして使う。
VQ/generative latent設計の参考にする。
CoSER-DiC semantic VQのdefault initializationには使わない。
```

---

### 8.5 lianqi1008/Control-GIC

Repository:

```text
https://github.com/lianqi1008/Control-GIC
```

Role:

```text
Variable-rate VQGAN generative compression baseline.
Reference for dynamic granularity and variable-rate control.
```

Usage policy:

```text
baselineとして使う。
variable-rate extension時の設計参考にする。
MVP-v0 core initializationには使わない。
```

---

### 8.6 NJUVISION/ResULIC

Repository:

```text
https://github.com/NJUVISION/ResULIC
```

Role:

```text
Semantic residual coding comparison.
Important conceptual baseline.
```

Usage policy:

```text
codeとpretrained weightsが完全に利用可能で再現可能ならbaseline評価する。
そうでなければrelated work / qualitative comparison / reported numbers扱いにする。
CoSER-DiC core initializationには使わない。
```

---

### 8.7 huai-chang/RDEIC

Repository:

```text
https://github.com/huai-chang/RDEIC
```

Role:

```text
Relay residual diffusion / extreme image compression comparison.
```

Usage policy:

```text
codeとpretrained weightsが十分に利用可能ならbaseline評価する。
そうでなければrelated work扱いにする。
CoSER-DiC core initializationには使わない。
```

---

## 9. Default pretrained policy

MVP-v0におけるpretrained policyは以下で固定する。

```text
Use pretrained CoD-Lite for diffusion reconstruction initialization.
Do not use pretrained RDVQ / GLC / Control-GIC as default CoSER semantic VQ initialization.
Do not use StableCodec as default CoSER decoder unless CoD-Lite integration fails.
Use CompressAI pretrained models as baselines only.
```

詳細:

| Component | Default source | Training policy |
|---|---|---|
| Semantic VQ encoder | CoSER-DiC | scratch training |
| Semantic VQ codebook | CoSER-DiC | scratch training |
| Semantic entropy model | CoSER-DiC | scratch training |
| Detail residual encoder | CoSER-DiC | scratch training |
| Detail residual entropy model | CoSER-DiC | scratch training |
| Auxiliary decoder | CoSER-DiC | scratch training |
| Diffusion decoder backbone | CoD-Lite pretrain | initialize, then train adapter / selected layers |
| CoSER conditioning adapter | CoSER-DiC | scratch training |
| RD baselines | CompressAI | official pretrained |
| VQ baselines | RDVQ / GLC / Control-GIC | official pretrained |
| One-step diffusion baselines | CoD-Lite / StableCodec / AEIC / OneDC | official pretrained |

---

## 10. MVP-v0 reference implementation policy

MVP-v0の実装方針は以下である。

```text
CoSER-DiC-MVP-v0:
  Semantic VQ encoder/tokenizer: scratch training
  Semantic entropy model: scratch training
  Detail residual encoder: scratch training
  Detail entropy model: scratch training
  Auxiliary decoder: scratch training
  Diffusion decoder backbone: initialize from CoD-Lite pretrain if feasible
  CoSER-specific conditioning adapter / FiLM / control branch: scratch training
  Actual bitstream coder: implemented and tested in CoSER-DiC
```

MVP-v0のforward path:

```text
x
  -> SemanticEncoder
  -> DifferentiableVQ
  -> semantic token stream
  -> SemanticAuxiliaryDecoder
  -> x_sem
  -> DetailResidualEncoder(x, x_sem, residual features)
  -> detail latent stream
  -> JointAuxiliaryDecoder(s_hat, d_hat)
  -> x_aux
  -> CoD-Lite-initialized diffusion decoder + CoSER adapter
  -> x_hat
```

MVP-v0のofficial evaluation path:

```text
x
  -> CoSERDiC.compress(x)
  -> actual bitstream bytes
  -> CoSERDiC.decompress(bitstream)
  -> x_hat
  -> metrics
```

---

## 11. Bitrate and side-information policy

論文に使う結果は、必ずactual bitstreamに基づく。

Required bpp calculation:

```text
actual_bpp = 8 * total_bitstream_bytes / (H * W)
```

Required bpp decomposition:

```text
total_actual_bpp
semantic_token_bpp
semantic_hyper_bpp
detail_latent_bpp
detail_hyper_bpp
header_bpp
optional_control_bpp
```

許可されるもの:

```text
estimated bpp for training diagnostics
estimated bpp for debugging
```

禁止されるもの:

```text
forward-only estimated bppをmain resultとして報告すること
original-derived masksをdecoderに無料で渡すこと
original-derived captionsをdecoderに無料で渡すこと
original-derived segmentation mapsをdecoderに無料で渡すこと
original-derived OCR mapsをdecoderに無料で渡すこと
original-derived importance mapsをdecoderに無料で渡すこと
```

これらをdecoderで使う場合は、必ずentropy codingしてbitstreamに含め、actual bppに計上する。

---

## 12. Baseline priority for MVP-v0

最初に走らせるbaselineは以下に絞る。

### Priority A: Must run first

```text
1. CompressAI Cheng2020 / hyperprior anchors
2. CoD-Lite rate checkpoints
3. RDVQ checkpoints
4. GLC pretrained
5. StableCodec checkpoints
```

### Priority B: Run next if time allows

```text
1. AEIC
2. Control-GIC
3. OneDC
4. RDEIC
5. ResULIC
```

### Classical codecs

```text
BPG
VVC/VTM
JPEG XL or AVIF if available
```

Main tableには、可能な限り同一pipelineで再評価した値を載せる。論文値引用は補助扱いにする。

---

## 13. Required ablations for first serious result

MVP-v0の最初のまとまった結果では、最低限以下のablationを実施する。

```text
A0: semantic-only reconstruction
A1: semantic VQ + auxiliary decoder, no detail branch
A2: semantic VQ + detail branch + auxiliary decoder, no diffusion
A3: semantic VQ + diffusion, no detail branch
A4: semantic VQ + detail branch + diffusion
A5: A4 without semantic-conditioned residual prior
A6: A4 using estimated bpp only as diagnostic, actual bpp as official result
```

このablationで示すべき因果:

```text
semantic-only preserves global structure but loses critical details.
detail residual improves faithfulness at controlled bitrate.
diffusion improves perceptual realism without excessive semantic drift.
semantic-conditioned residual entropy model improves rate efficiency or critical-detail preservation.
actual bpp remains reasonably close to estimated bpp.
```

---

## 14. Success criteria for MVP-v0

MVP-v0完了条件:

```text
1. compress() / decompress() APIが動作する
2. actual bitstreamから復元できる
3. decoderがoriginal imageにアクセスしない
4. estimated bppとactual bppを両方reportできる
5. 0.01 / 0.03 / 0.05 bpp付近で結果がある
6. Kodak / CLIC / DIV2Kで評価済み
7. semantic-only / no-residual / no-diffusion ablationがある
8. 少なくとも2つ以上の直接競合baselineと比較済み
9. visual comparisonがある
10. semantic/detail bitrate decompositionがある
```

MVP-v0が完了するまで、VLM-DPO、OCR loss、face identity loss、adaptive quantization、variable-rate extensionには進まない。

---

## 15. Unit tests / sanity checks

実装前半で必ず以下のテストを作る。

```text
1. semantic token round-trip test
2. detail latent round-trip test
3. bitstream round-trip test
4. estimated vs actual bpp test
5. padding/cropping correctness test
6. decoder leakage prevention test
7. deterministic decompression test
```

重要:

```text
encoded semantic tokens == decoded semantic tokens
encoded detail latents == decoded detail latents
same bitstream -> same x_hat
metric is computed only on original unpadded region
decompress() must not access original image or encoder-side-only tensors
```

---

## 16. Immediate action items

研究を本格的に進める前に、以下を実施する。

```text
1. Add this document to:
   docs/research/design_decisions/000_external_repos_and_pretrained_policy_mvp.md

2. Register external repositories in:
   docs/research/baseline_registry.md

3. Add config stubs for:
   configs/baselines/compressai_cheng2020.yaml
   configs/baselines/compressai_hyperprior.yaml
   configs/baselines/cod_lite_0039.yaml
   configs/baselines/cod_lite_0078.yaml
   configs/baselines/cod_lite_0156.yaml
   configs/baselines/cod_lite_0312.yaml
   configs/baselines/rdvq.yaml
   configs/baselines/glc.yaml
   configs/baselines/stablecodec.yaml

4. Add pretrained checkpoint paths to:
   configs/paths/pretrained.yaml

5. Ensure the following directories are gitignored:
   external/repos/
   external/pretrained/
   checkpoints/
   results/
   bitstreams/
   artifacts/
   wandb/

6. Implement or verify unit tests for:
   semantic token round trip
   detail latent round trip
   bitstream round trip
   estimated vs actual bpp
   padding/cropping correctness
   decoder leakage prevention
```

---

## 17. Final instruction

CoSER-DiCは新規codecである。

External pretrained models are tools, not the method.

MVP-v0では、以下の解釈で進める。

```text
CoSER-DiC-MVP-v0 uses:
  - scratch-trained semantic VQ
  - scratch-trained detail residual branch
  - scratch-trained entropy models
  - scratch-trained auxiliary decoder
  - CoD-Lite-pretrained diffusion backbone if feasible
  - CoSER-specific conditioning adapter
  - actual entropy-coded bitstream

RDVQ, GLC, Control-GIC, StableCodec, AEIC, OneDC, CompressAI, ResULIC, and RDEIC are:
  - baselines
  - implementation references
  - pretrained initialization candidates only where explicitly stated
  - optional ablation sources after MVP is stable
```

最初の研究目標は、full versionの全要素を入れることではない。

最初の研究目標は、以下を実証することである。

```text
Semantic VQだけではglobal structureは保てるがcritical detailsが不足する。
Detail residual branchはcontrolled bitrateでfaithfulnessを改善する。
Diffusion decoderはtexture realismを改善する。
Semantic-conditioned residual entropy modelはdetail residualのrate efficiencyを改善する。
この分離設計により、extreme-low / low bitrateでR-D-P trade-offが改善する。
```

この因果をMVP-v0で示せるまでは、full versionの拡張には進まない。
