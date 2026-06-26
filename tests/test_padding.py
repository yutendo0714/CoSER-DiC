import torch

from coserdic.utils import crop_to_shape, pad_to_multiple


def test_pad_and_crop_chw() -> None:
    x = torch.rand(3, 67, 65)
    padded, shape = pad_to_multiple(x, multiple=64)
    assert shape == (67, 65)
    assert padded.shape[-2:] == (128, 128)
    restored = crop_to_shape(padded, shape)
    assert torch.equal(restored, x)


def test_pad_noop_when_aligned() -> None:
    x = torch.rand(1, 3, 64, 128)
    padded, shape = pad_to_multiple(x, multiple=64)
    assert padded is x
    assert shape == (64, 128)

