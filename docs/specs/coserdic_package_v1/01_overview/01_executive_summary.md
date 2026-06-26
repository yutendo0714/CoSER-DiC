# Executive Summary

## 提案手法名

**CoSER-DiC: Compression-Oriented Semantic-Residual Diffusion Codec**

論文タイトル候補:

> Entropy-Constrained Semantic-Residual Diffusion Codec for Ultra-Low-Rate Generative Image Compression

---

## 問題設定

極低bppの画像圧縮では、従来のRate-Distortion最適化codecは構造をある程度保てる一方で、復元がぼやける。GANやdiffusionを使う生成圧縮は自然なtextureを生成できるが、原画像に存在しない細部をhallucinateする危険がある。

0.003〜0.05 bppでは、画像の全情報を忠実にbitstreamへ送ることは不可能である。したがって、どの情報をbitで送るか、どの情報を生成priorに任せるかを明示的に設計する必要がある。

---

## 中心仮説

> Extreme-low-bitrate generative image compression benefits from separating semantic structure, detail-critical residuals, and stochastic texture generation under entropy-constrained coding.

日本語では以下である。

> 極低bpp生成圧縮では、semantic structure、detail-critical residual、stochastic textureを単一latentに混在させることが非効率である。Semantic VQ token、semantic-conditioned detail residual、compression-oriented diffusion decoderに分離して最適化することで、同じbitrateでよりfaithfulかつrealisticな再構成が可能になる。

---

## CoSER-DiCの役割分担

| 要素 | 送る/生成する情報 | 役割 |
|---|---|---|
| Semantic VQ token | 物体配置、scene layout、主要構造、大域色 | 少ないbitで意味構造を保持 |
| Detail residual latent | 顔、文字、小物、境界、identity、局所補正 | hallucinationを抑えfaithfulnessを補強 |
| Auxiliary decoder | 粗い復元 | diffusion decoderのfidelity anchor |
| Compression-oriented diffusion decoder | texture、高周波、自然性 | bitで送らないstochastic detailを生成 |
| VLM / preference tuning | 人間選好・hallucination抑制 | Full版のalignment補助 |

---

## MVPで狙う成果

MVPでは、以下を示す。

1. Semantic VQだけでは自然性またはfaithfulnessが不足する。
2. Detail residual branchを加えると、同じbpp付近でLPIPS/DISTSとcritical detailが改善する。
3. One-step diffusion decoderを加えると、FID/KIDとperceptual qualityが改善する。
4. Actual bitstream込みで、既存のGLC/RDVQ/StableCodec/ResULIC/RDEIC/CoD-Lite系と比較できる。

---

## 主戦場

```text
bpp: 0.003, 0.005, 0.01, 0.02, 0.03, 0.05
primary metrics: LPIPS, DISTS, FID, KID, actual bpp
secondary metrics: PSNR, MS-SSIM, CLIP/DINO similarity, OCR, face identity, runtime
```

PSNR SOTAを狙う必要はない。ただし、PSNRが極端に悪化すると「見た目だけのcodec」と批判されるため、distortion下限は維持する。

---

## 投稿可能性の条件

国際会議投稿候補とするには、最低限以下が必要。

```text
[ ] LPIPSまたはDISTSで主要生成系baselineを上回る
[ ] FID/KIDが同等以上
[ ] actual bpp込みで比較している
[ ] semantic faithfulness指標で改善がある
[ ] human 2AFCまたはVLM-assisted評価で補強できる
[ ] ablationで各moduleの役割が明確
[ ] baselineが同一評価pipelineで再評価されている
```
