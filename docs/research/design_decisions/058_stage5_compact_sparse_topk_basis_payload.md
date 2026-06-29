# 058 Stage 5 Compact Sparse Top-k Basis Payload

Date: 2026-06-29

## Decision

For fixed-bit sparse top-k basis controls, pack the selected basis indices and
their quantized coefficient values into one compact fixed-bit payload stream.

This replaces the previous fixed-bit simulation that encoded:

```text
top-k index stream
+ top-k value stream
```

as two independently byte-rounded streams.

The Huffman variant remains two entropy-coded streams because its index and
value payload lengths are image-dependent and are already counted exactly.

## Motivation

Stage 5 counted controls are intentionally tiny. At these rates, byte rounding
can dominate the actual payload bpp.

Example:

```text
candidate_components = 3
selected_components = 1
index_bits = 2
value_bits = 2
```

The true fixed payload is:

```text
2 + 2 = 4 bits -> 1 byte
```

Encoding index and value as separate byte-rounded streams would count:

```text
1 byte + 1 byte = 2 bytes
```

That is a real bpp over-count for this fixed-bit codec family. Since the decoder
knows the candidate count, selected count, and value quantizer from fixed model
state, a single packed stream is deterministic and does not add hidden side
information.

## Bitstream Rule

For fixed-bit top-k basis control:

```text
payload_bits =
  selected_components * ceil(log2(candidate_components))
  + selected_components * value_bits

payload_bytes = ceil(payload_bits / 8)
```

These bytes are included in:

```text
actual_payload_bpp / paper_bpp
```

For hybrid selectors with exact selector bits:

```text
hybrid_payload_bytes =
  ceil((selector_bits + payload_bits) / 8)
```

with a guard that never reports fewer bytes than the simulated compact payload.

## Implementation

Added:

```text
SparseControlBasisCode.encode_compact
SparseControlBasisCode.decode_compact
SparseControlBasisCode.encoded_compact_num_bits
SparseControlBasisCode.encoded_compact_num_bytes
```

Updated:

```text
scripts/eval_stage4_cod_lite_adapter.py
  grouped_condition_basis_control fixed top-k path
  hybrid-basis exact bit accounting

scripts/sweep_stage5_counted_control.py
  dry-run control_bytes for basis / affine_basis / hybrid_basis top-k fixed_bits
```

Huffman sparse top-k remains:

```text
index_huffman_payload + value_huffman_payload
```

because those are real variable-length entropy-coded payloads.

## Verification

CPU verification passed:

```bash
PYTHONPATH=/workspace/CoSER-DiC .venv/bin/pytest \
  tests/test_control_grid.py \
  tests/test_eval_stage4_cod_lite_adapter_control.py \
  tests/test_sweep_stage5_counted_control.py -q
```

Result:

```text
67 passed
```

Full test suite:

```bash
PYTHONPATH=/workspace/CoSER-DiC .venv/bin/pytest tests -q
```

Result:

```text
263 passed, 4 warnings
```

## Research Interpretation

This is not a new quality trick and not RGB postprocessing. It is cleaner
actual-bitstream accounting for a counted control stream.

It matters for Stage 5 because low-rate basis/top-k/hybrid candidates are
intended to become BD-rate curve points. Over-counting fixed-bit control bytes
would unfairly penalize the CoSER control stream, while under-counting would
invalidate the paper metric. Compact packing is the correct middle ground.
