# CoSER-DiC Complete Research Handoff Package v1.0

作成日: 2026-06-26  
対象: Learned Image Compression / Generative Compression / Ultra-low-bitrate Compression を専門とする研究者・実装担当者  
目的: **CoSER-DiC: Compression-Oriented Semantic-Residual Diffusion Codec** をゼロから実装し、国際会議投稿を見据えて研究を開始・推進・評価・論文化するための完全仕様パッケージ。

---

## 研究者への最短指示

このZIPを受け取った研究者には、まず以下の順に読ませてください。

1. `01_overview/00_researcher_handoff_letter.md`
2. `01_overview/01_executive_summary.md`
3. `02_method/00_non_negotiable_core.md`
4. `02_method/01_mvp_reference_architecture.md`
5. `03_implementation/00_repository_and_api_spec.md`
6. `03_implementation/01_bitstream_actual_bpp_spec.md`
7. `04_training/00_training_protocol.md`
8. `05_evaluation/00_evaluation_protocol.md`
9. `06_baselines/00_baseline_protocol.md`
10. `07_acceptance/00_acceptance_criteria.md`

これらを読めば、研究者はMVP実装、baseline再現、評価、アブレーション、論文化までの進め方を把握できる構成になっています。

---

## 本パッケージの思想

CoSER-DiCは、単に既存のVQ、residual coding、diffusion decoder、VLM preference trainingを足し合わせる提案ではありません。

中心仮説は次です。

> **極低bpp生成圧縮では、semantic structure、detail-critical residual、stochastic textureを単一latentに混在させることが非効率である。Semantic VQ token、semantic-conditioned detail residual、compression-oriented one-step diffusion decoderに分離してentropy制約下で最適化することで、faithfulnessとrealismの両方を改善できる。**

研究者が勝手に違う方向へ進まないよう、MVP v0では以下を固定します。

```text
RD-aware Semantic VQ
+
Semantic-conditioned Detail Residual Coding
+
Auxiliary Reconstruction
+
Compression-oriented One-step Diffusion Decoder
+
Actual Entropy Coding / Actual bpp reporting
```

これらを外した実装はCoSER-DiCとは呼ばないでください。

---

## ディレクトリ構成

```text
01_overview/          研究者への伝達文、全体概要、研究仮説
02_method/            提案手法の核、MVP固定仕様、full拡張仕様、数式
03_implementation/    実装API、bitstream、repo構成、テスト、コード骨格
04_training/          stage-wise training、loss、rate control、debug方針
05_evaluation/        evaluation protocol、metrics、human/VLM評価
06_baselines/         baseline再現プロトコル、比較優先順位、source inventory
07_acceptance/        成功条件、納品物、受け入れ基準
08_project_ops/       milestone、ログ、再現性、リスク管理
09_paper/             論文ストーリー、method節、claim設計、図表計画
10_templates/         実験ログ・実験カード・ablation表テンプレート
11_configs/           YAML形式の初期config案
12_pseudocode/        PyTorch API骨格・training loop・bitstream疑似コード
13_references/        参考文献・公開実装・ライセンス注意点
```

---

## 実装開始時の最重要ルール

1. **estimated bppだけで結果を出さない。** 必ずactual arithmetic-coded bitstreamに基づくactual bppを報告する。
2. **decoderはbitstream以外のoriginal-derived informationを使ってはいけない。** caption、segmentation map、OCR mask、face mask、importance mapを使うなら必ずbitstreamに含め、bppに計上する。
3. **MVPでは複雑な拡張を入れない。** VLM-DPO、OCR loss、face identity loss、full autoregressive prior、Mamba prior、multi-step diffusionはMVP後。
4. **PSNR SOTAを主張の中心にしない。** 主戦場は0.003〜0.05 bppのLPIPS、DISTS、FID/KID、semantic faithfulness、runtime、human preference。
5. **成功例だけでなく失敗例を残す。** 生成圧縮はhallucinationが本質的なリスクであり、failure galleryは論文化に必要。

---

## 期待される最初の成果物

MVP完了と認める最低条件は以下です。

```text
[ ] compress/decompress APIが動作する
[ ] bitstreamだけから復元できる
[ ] estimated bppとactual bppを両方reportできる
[ ] 0.01 / 0.03 / 0.05 bpp付近の結果がある
[ ] Kodak / CLIC / DIV2Kで評価済み
[ ] semantic-only / no-residual / no-diffusion ablationがある
[ ] 直接競合baselineを最低2つ以上同一pipelineで再評価している
[ ] visual comparisonとfailure galleryがある
```

---

## 注意

本パッケージは研究仕様書であり、実装済みの完全コードではありません。ただし、実装者が迷わないようにreference architecture、API、YAML config、pseudo-code、評価ルール、acceptance criteriaを含めています。

公開実装や論文のURLは `13_references/` にまとめています。利用時は必ず各リポジトリのライセンス、checkpoint利用条件、dataset利用条件を研究者側で再確認してください。
