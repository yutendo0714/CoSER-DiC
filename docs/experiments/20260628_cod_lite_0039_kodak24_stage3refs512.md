# 20260628_cod_lite_0039_kodak24_stage3refs512

Date: 2026-06-28T16:49:56

## Command

```bash
scripts/eval_cod_lite_official_baseline.py --checkpoint external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0039.pt --config external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0039.yaml --input-dir results/baselines/cod_lite_official/20260628_kodak24_stage3refs512_input --run-name 20260628_cod_lite_0039_kodak24_stage3refs512 --fid-patch-size 64 --fid-patch-num 2
```

## Summary

- count: 24
- actual_payload_bpp_mean: 0.00390625
- cod_payload_bpp_mean: 0.00390625
- codec_file_bpp_mean: 0.0040283203125
- psnr_mean: 19.030364672342937
- ms_ssim_mean: 0.5700366633633772
- lpips_alex_mean: 0.33898304402828217
- dists_mean: 0.20890280604362488
- fid: 51.218963623046875
- fid_patch_size: 64
- fid_patch_num: 2
- payload_policy: CoD-Lite .cod size minus 4-byte width/height header; fixed model/codebook weights excluded.
- checkpoint: external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0039.pt
- config: external/pretrained/CoD_Lite/CoD_Lite_bpp_0_0039.yaml
- input_dir: results/baselines/cod_lite_official/20260628_kodak24_stage3refs512_input
- reconstruction_dir: results/baselines/cod_lite_official/20260628_cod_lite_0039_kodak24_stage3refs512/reconstructions
- bitstream_dir: results/baselines/cod_lite_official/20260628_cod_lite_0039_kodak24_stage3refs512/bitstreams
- main_bpp_metric: actual_payload_bpp_mean

## Artifacts

- summary: `results/baselines/cod_lite_official/20260628_cod_lite_0039_kodak24_stage3refs512/summary.json`
- per_image: `results/baselines/cod_lite_official/20260628_cod_lite_0039_kodak24_stage3refs512/per_image_metrics.jsonl`
- bitstreams: `results/baselines/cod_lite_official/20260628_cod_lite_0039_kodak24_stage3refs512/bitstreams`
- reconstructions: `results/baselines/cod_lite_official/20260628_cod_lite_0039_kodak24_stage3refs512/reconstructions`
