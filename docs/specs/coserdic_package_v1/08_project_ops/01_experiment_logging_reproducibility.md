# Experiment Logging and Reproducibility

全実験で以下を保存する。

---

## Run metadata

```yaml
run:
  run_id:
  date:
  git_commit:
  config_file:
  config_hash:
  seed:
  dataset:
  train_split:
  eval_split:
  checkpoint:
  model_variant:
  target_bpp:
  estimated_bpp:
  actual_bpp:
  semantic_bpp:
  detail_bpp:
  hyper_bpp:
  header_bpp:
  metrics:
    psnr_rgb:
    psnr_y:
    ms_ssim:
    lpips:
    dists:
    fid:
    kid:
  runtime:
    encode_time:
    decode_time:
    arithmetic_coding_time:
    gpu_memory:
  notes:
```

---

## 保存物

```text
config
checkpoint
logs
metric json
bitstreams
reconstructed images
visual crops
failure cases
```

---

## Naming convention

```text
{date}_{method}_{stage}_{dataset}_{rate_level}_{run_id}
```

例:

```text
20260626_coserdic_stage3_kodak_003bpp_ab12cd
```

---

## Reproducibility command

各main resultには、再実行用コマンドを保存する。

```text
python eval/eval_rdp.py --config configs/eval_kodak.yaml --checkpoint ...
```

---

## Config hash

configを変更したら新しいrun_idにする。小さなloss weight変更でも別runとして扱う。

---

## Dataset notes

datasetは以下を記録する。

```text
source
version/download date
split
preprocessing
excluded images if any
```

---

## Metric implementation version

特に以下はversion差が出やすい。

```text
LPIPS
DISTS
FID/KID
MS-SSIM
YCbCr conversion
```

必ずcommit/hashまたはpackage versionを記録する。
