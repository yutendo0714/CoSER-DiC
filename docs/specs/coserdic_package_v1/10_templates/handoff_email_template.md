# Handoff Email Template

件名: CoSER-DiC研究実装パッケージの共有とMVP実装依頼

本文:

CoSER-DiC: Compression-Oriented Semantic-Residual Diffusion Codec の研究実装パッケージを共有します。

まず `README.md` と `01_overview/00_researcher_handoff_letter.md` を読み、その後 `02_method/01_mvp_reference_architecture.md`、`03_implementation/01_bitstream_actual_bpp_spec.md`、`04_training/00_training_protocol.md`、`05_evaluation/00_evaluation_protocol.md` を確認してください。

最初のMVPでは、RD-aware Semantic VQ、Semantic-conditioned Detail Residual Coding、Auxiliary Decoder、One-step Diffusion Decoder、Actual Entropy Codingのみを実装対象にしてください。VLM-DPO、OCR loss、face identity loss、adaptive quantization、full autoregressive priorはMVP後の拡張です。

重要な制約として、評価はestimated bppではなくactual arithmetic-coded bitstreamに基づくactual bppで行ってください。また、decoderはbitstream以外のoriginal-derived informationにアクセスしてはいけません。

MVP完了条件は `07_acceptance/00_acceptance_criteria.md` に定義しています。まず0.01 / 0.03 / 0.05 bpp付近でKodak、CLIC、DIV2Kに対して評価し、semantic-only、no-residual、no-diffusionのアブレーションを出してください。

不明点が出た場合は、仕様の自由解釈で進める前に、どの仕様に関する判断かを明記して相談してください。
