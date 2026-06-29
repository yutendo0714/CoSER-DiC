# Decision 052: Stage 5 Hybrid Selector Bit Accounting

Date: 2026-06-29 JST

## Status

Implemented in evaluation/planning code. CPU tests passed.

GPU execution has not been launched because CUDA/NVML is currently unavailable.

## Motivation

Stage 5 hybrid condition control transmits an image-specific mode selector.
That selector is required by the decoder, so it must be counted in
`actual_payload_bpp`.

The earlier implementation used the conservative rule:

```text
selected_payload_bytes = selected_control_payload_bytes + selector_bytes
```

with `selector_bytes = 1` by default.

This is safe, but it hides two separate accounting facts:

```text
selector bit count
payload byte rounding
```

For paper-quality accounting and future compact bitstream work, those should be
tracked explicitly.

## Decision

Add optional selector-bit accounting:

```text
--control-hybrid-selector-bytes 1
--control-hybrid-selector-bits <0|2|3|...>
```

Rules:

```text
selector_bits = 0:
  keep legacy byte-granular accounting

selector_bits > 0 and fixed payload bits are known:
  count ceil((payload_bits + selector_bits) / 8)
  but never below the already simulated payload byte count

selector_bits > 0 and payload bits are unknown:
  fall back to payload_bytes + max(selector_bytes, ceil(selector_bits / 8))
```

This avoids undercounting variable-length Huffman/top-k streams while allowing
fixed-bit streams to use exact packed-stream accounting.

## Implementation

```text
src/coserdic/entropy/control_grid.py
  encoded_num_bits for fixed control codecs

scripts/eval_stage4_cod_lite_adapter.py
  --control-hybrid-selector-bits
  payload_bytes_with_hybrid_selector(...)
  exact selector accounting in condition-L1 and image-RDO hybrid selectors

scripts/sweep_stage5_counted_control.py
  selector_bits=<int> in setting strings
  --override-hybrid-selector-bits
  dry-run control_bpp reflects the same selector accounting as eval
```

The default remains `selector_bits=0`, so existing plans keep their
conservative one-byte selector behavior.

## Current Audit

A selector-bits dry-run plan was generated:

```text
results/stage5_counted_control/20260629_detailfilm_hybrid_selectorbits_limit64_sweep_plan.json
results/stage5_counted_control/20260629_detailfilm_hybrid_selectorbits_limit64_sweep_plan.sh
```

For the current main 4-bit hybrid settings, `selector_bits=2` did not reduce
planned `control_bytes` because the candidate payloads are already byte
aligned. Example:

```text
old: selector_bytes=1, selector_bits=0, control_bytes=25
new: selector_bytes=1, selector_bits=2, control_bytes=25
```

This means we are not getting a cosmetic bpp reduction from selector packing in
the current settings. It is still useful metadata for future non-byte-aligned
control streams.

## Verification

Passed:

```bash
PYTHONPATH=/workspace/CoSER-DiC .venv/bin/pytest \
  tests/test_eval_stage4_cod_lite_adapter_control.py \
  tests/test_sweep_stage5_counted_control.py -q
```

Result:

```text
32 passed
```

`git diff --check` also passed.

## Promotion Rule

Selector-bit accounting alone is not a Stage 5 improvement.

Do not promote a candidate because `selector_bits` is present. Promote only if
decoded-image metrics improve under actual payload accounting and fidelity
guards.
