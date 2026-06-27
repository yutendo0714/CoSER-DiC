# Decision 010: Bpp Reporting Policy

Date: 2026-06-27

## Decision

CoSER-DiC paper tables use `actual_payload_bpp` as the primary LIC comparison rate.

```text
actual_payload_bpp = paper_bpp
                   = 8 * sum(bytes of entropy-coded payload streams required by decoder)
                     / (original_H * original_W)
```

This matches the common CompressAI-style learned image compression convention:
count the transmitted entropy-coded strings, not ordinary container/debug metadata.

## Counted in actual_payload_bpp

- Semantic token stream.
- Semantic hyper or side stream, if transmitted.
- Detail residual latent stream.
- Detail hyper or side stream, if transmitted.
- Control stream, if transmitted.
- Any image-specific prompt, mask, table, map, seed, or side information that the decoder cannot reproduce without transmission.

For the current Core MVP Stage3 uniform residual codec, this is:

```text
actual_payload_bpp = semantic_payload_bpp + detail_payload_bpp
```

If a residual codec transmits an explicit mode bit, that bit is embedded in the
detail payload and counted. The current active semantic-position-left residual
codec does not transmit a mode bit.

## Excluded from paper_bpp

- `out_enc["shape"]` style decode shape metadata.
- Original image size, padded size, stream lengths, magic number, checksum, codec ids, and compact/JSON container header.
- Debug JSON and experiment logs.
- Model weights, fixed pretrained decoder weights, fixed VQ codebook, fixed entropy prior, and fixed Huffman tables trained offline.

These are not ignored forever; they are reported separately for engineering audits.

## Metric Names

- `estimated_bpp`: likelihood/model-estimated bitrate. Training diagnostic only.
- `actual_payload_bpp`: entropy-coded payload streams only. Main LIC paper metric.
- `paper_bpp`: alias for `actual_payload_bpp`.
- `debug_full_stream_bpp`: payload plus CoSER container/header/checksum. Roundtrip and development audit only.
- `codec_file_bpp`: optional production file-size metric if/when a compact file format is finalized.

Legacy names remain for backwards compatibility:

- `total_payload_bpp_mean` is equivalent to `actual_payload_bpp_mean` for current Stage3 payload-only evaluations.
- `stage3_full_stream_bpp_mean` is equivalent to `debug_full_stream_bpp_mean`, not the paper metric.
- Stage1/Stage2 legacy `actual_bpp_mean` refers to full CoSER stream bpp in old scripts/logs; new runs also emit `actual_payload_bpp_mean`.

## Implementation

`CoSERBitstream` exposes explicit helpers:

- `actual_payload_bpp(...)`
- `debug_full_stream_bpp(...)`
- `actual_bpp(...)` as a backwards-compatible alias for full byte-count bpp

Evaluation scripts now emit `main_bpp_metric`, `paper_bpp_metric`, and `debug_bpp_metric` in summaries to prevent ambiguous tables.

## Current Core MVP Reading

For the 2026-06-27 Stage3 semantic-position-left residual anchors, use
`total_payload_bpp_mean` / `actual_payload_bpp_mean` in paper-style
comparisons:

- b4 low-rate anchor: Kodak `0.015828`, DIV2K100 `0.015658`, CLIC41 `0.014708`, CLIC64 `0.014931`.
- b5 quality anchor: Kodak `0.018473`, DIV2K100 `0.018390`, CLIC41 `0.017257`, CLIC64 `0.017487`.

The corresponding compact-v3 CRC32 full stream values are `debug_full_stream_bpp`, useful for checking implementation overhead but not the main LIC comparison rate.
