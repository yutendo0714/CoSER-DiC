import torch

from coserdic.models import ResidualDetailAutoEncoder, ResidualDetailConfig, UniformScalarQuantizer


def test_uniform_scalar_quantizer_bounds_and_shape() -> None:
    quantizer = UniformScalarQuantizer(bits=4, value_range=0.25)
    x = torch.tensor([[-1.0, -0.25, 0.0, 0.25, 1.0]])
    y, codes = quantizer(x, training_noise=False)
    assert y.shape == x.shape
    assert codes.shape == x.shape
    assert int(codes.min()) == 0
    assert int(codes.max()) == 15
    assert float(y.min()) >= -0.25
    assert float(y.max()) <= 0.25


def test_residual_detail_autoencoder_forward_shapes() -> None:
    cfg = ResidualDetailConfig(
        semantic_channels=8,
        base_channels=16,
        hidden_channels=32,
        detail_channels=3,
        num_res_blocks=1,
        quant_bits=5,
        value_range=0.25,
    )
    model = ResidualDetailAutoEncoder(cfg)
    x = torch.rand(2, 3, 64, 64)
    x_sem = torch.rand(2, 3, 64, 64)
    semantic = torch.rand(2, 8, 2, 2)
    out = model(x, x_sem, semantic, training_noise=False)
    assert out["x_aux"].shape == x.shape
    assert out["residual_pred"].shape == x.shape
    assert out["h_detail"].shape == (2, 3, 2, 2)
    assert out["detail_latent"].shape == (2, 3, 2, 2)
    assert out["detail_codes"].shape == (2, 3, 2, 2)
    assert int(out["detail_codes"].min()) >= 0
    assert int(out["detail_codes"].max()) < 32


def test_residual_detail_eval_quantization_is_deterministic() -> None:
    cfg = ResidualDetailConfig(
        semantic_channels=8,
        base_channels=16,
        hidden_channels=32,
        detail_channels=2,
        num_res_blocks=1,
    )
    model = ResidualDetailAutoEncoder(cfg).eval()
    x = torch.rand(1, 3, 64, 64)
    x_sem = torch.rand(1, 3, 64, 64)
    semantic = torch.rand(1, 8, 2, 2)
    out_a = model(x, x_sem, semantic, training_noise=False)
    out_b = model(x, x_sem, semantic, training_noise=False)
    assert torch.equal(out_a["detail_codes"], out_b["detail_codes"])
    assert torch.allclose(out_a["x_aux"], out_b["x_aux"])
