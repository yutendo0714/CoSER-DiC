# Repository and API Specification

## 推奨リポジトリ構成

```text
coserdic/
  README.md
  LICENSE
  configs/
    train_stage1_semantic_vq.yaml
    train_stage2_entropy.yaml
    train_stage3_residual.yaml
    train_stage4_diffusion.yaml
    train_stage5_joint.yaml
    train_variable_rate.yaml
    eval_kodak.yaml
    eval_clic.yaml

  coserdic/
    models/
      semantic_encoder.py
      vq.py
      semantic_entropy.py
      detail_encoder.py
      detail_entropy.py
      aux_decoder.py
      diffusion_decoder.py
      rate_conditioner.py
      coserdic.py

    entropy/
      arithmetic_coder.py
      categorical_model.py
      gaussian_conditional.py
      bitstream.py

    losses/
      distortion.py
      perceptual.py
      adversarial.py
      semantic.py
      preference.py

    datasets/
      image_folder.py
      clic.py
      div2k.py
      kodak.py
      transforms.py

    metrics/
      psnr.py
      msssim.py
      lpips.py
      dists.py
      fid.py
      kid.py
      semantic.py
      ocr.py
      face.py
      runtime.py

    train/
      train_stage1.py
      train_stage2.py
      train_stage3.py
      train_stage4.py
      train_joint.py

    eval/
      eval_rd.py
      eval_rdp.py
      eval_runtime.py
      eval_human_pairs.py
      make_visuals.py

    utils/
      logging.py
      checkpoint.py
      distributed.py
      image_io.py
      padding.py
      reproducibility.py

  scripts/
    prepare_datasets.sh
    train_stage1.sh
    train_stage2.sh
    train_stage3.sh
    train_stage4.sh
    train_joint.sh
    eval_all.sh
    compress_image.py
    decompress_image.py
```

---

## CoSERDiC API

研究者には以下のinterfaceを実装してもらう。

```python
class CoSERDiC(nn.Module):
    def forward_train(
        self,
        x,
        target_bpp=None,
        rate_level=None,
        perception_level=None,
        stage=None,
    ):
        """
        Returns:
            dict with:
                x_hat
                x_aux
                x_sem
                rate_s
                rate_d
                rate_total
                semantic_tokens
                detail_latents
                losses
                diagnostics
        """

    def compress(
        self,
        x,
        target_bpp=None,
        rate_level=None,
        perception_level=None,
    ):
        """
        Returns:
            bitstream bytes
            metadata
        """

    def decompress(self, bitstream):
        """
        Returns:
            reconstructed image x_hat
        """

    def estimate_bpp(self, x, target_bpp=None, rate_level=None):
        """
        Estimated bpp before actual arithmetic coding.
        Used only for training diagnostics.
        """
```

---

## Module構成

```python
class SemanticEncoder(nn.Module):
    pass

class DifferentiableVQ(nn.Module):
    pass

class SemanticEntropyModel(nn.Module):
    pass

class DetailResidualEncoder(nn.Module):
    pass

class DetailEntropyModel(nn.Module):
    pass

class AuxiliaryDecoder(nn.Module):
    pass

class DiffusionDecoder(nn.Module):
    pass

class RateConditioner(nn.Module):
    pass

class CoSERBitstreamCoder:
    pass
```

---

## compress / decompressの厳守事項

```text
compress()ではactual arithmetic codingを行う。
decompress()はbitstreamだけから復元できる。
original image由来の情報をdecoderに渡してはいけない。
estimated bppとactual bppを必ず両方reportする。
```

---

## Training forwardとActual codingの分離

training中はsoft quantization、noise relaxation、estimated bppを用いてよい。

ただし、評価結果として報告する値は必ず以下を通す。

```text
x -> compress() -> bitstream -> decompress() -> x_hat
```

forward-only evaluationはdebug用途に限定する。

---

## Determinism

diffusion decoderはMVPではdeterministic one-stepを基本にする。

stochastic samplingを使う場合:

```text
seedをheaderに入れる
同一bitstreamから同一復元が得られることをテストする
GPU非決定性の影響を記録する
```

---

## Unit tests

実装初期から以下のunit testを用意する。

```text
1. semantic token encode/decode一致
2. detail latent encode/decode一致
3. bitstream round-trip
4. padding/cropping consistency
5. estimated vs actual bpp差分
6. decoder leakage test
7. deterministic decompression test
```
