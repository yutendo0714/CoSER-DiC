from __future__ import annotations

import pytest
import torch
from torch import nn

from scripts.inspect_cod_lite_backbone_params import (
    choose_device,
    compile_patterns,
    direct_parameter_count,
    lora_param_count,
    module_row,
)


def test_compile_patterns_splits_commas() -> None:
    patterns = compile_patterns([r"decoder,attn", r"mlp"])

    assert [pattern.pattern for pattern in patterns] == ["decoder", "attn", "mlp"]


def test_choose_device_auto_returns_available_backend() -> None:
    assert choose_device("auto") in {"cpu", "cuda"}
    assert choose_device("cpu") == "cpu"
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA is not visible"):
            choose_device("cuda")


def test_lora_linear_module_row() -> None:
    module = nn.Linear(8, 4, bias=True)
    row = module_row("decoder.proj", module, lora_rank=2)

    assert row is not None
    assert row["name"] == "decoder.proj"
    assert row["type"] == "Linear"
    assert row["base_param_count"] == direct_parameter_count(module)
    assert row["lora_supported"] is True
    assert row["lora_param_count"] == 2 * (8 + 4)


def test_lora_conv2d_module_row() -> None:
    module = nn.Conv2d(3, 5, kernel_size=3, padding=1, bias=True)
    row = module_row("decoder.conv", module, lora_rank=4)

    assert row is not None
    assert row["type"] == "Conv2d"
    assert row["lora_supported"] is True
    assert row["lora_param_count"] == 4 * (3 * 3 * 3 + 5)
    assert lora_param_count(module, rank=4) == row["lora_param_count"]


def test_grouped_conv2d_is_marked_unsupported_for_lora() -> None:
    module = nn.Conv2d(4, 4, kernel_size=3, padding=1, groups=2)
    row = module_row("decoder.grouped", module, lora_rank=4)

    assert row is not None
    assert row["lora_supported"] is False
    assert row["lora_param_count"] == 0
