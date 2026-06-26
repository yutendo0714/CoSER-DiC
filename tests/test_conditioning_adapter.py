import torch

from coserdic.models import CoSERConditioningAdapter, ConditioningAdapterConfig, CoSERDiffusionDecoder


def test_conditioning_adapter_shapes() -> None:
    cfg = ConditioningAdapterConfig(
        semantic_channels=8,
        detail_channels=4,
        hidden_channels=16,
        rate_levels=3,
        rate_embed_dim=6,
    )
    adapter = CoSERConditioningAdapter(cfg)
    s_hat = torch.randn(2, 8, 4, 5)
    d_hat = torch.randn(2, 4, 4, 5)
    out = adapter(s_hat, d_hat, rate_level=1)
    assert out["condition"].shape == (2, 16, 4, 5)
    assert out["film_scale"].shape == (2, 16, 1, 1)
    assert out["film_shift"].shape == (2, 16, 1, 1)


def test_diffusion_decoder_debug_identity_is_explicit() -> None:
    cfg = ConditioningAdapterConfig(
        semantic_channels=8,
        detail_channels=4,
        hidden_channels=16,
        rate_levels=3,
        rate_embed_dim=6,
    )
    decoder = CoSERDiffusionDecoder(adapter=CoSERConditioningAdapter(cfg))
    x_aux = torch.rand(2, 3, 32, 32)
    s_hat = torch.randn(2, 8, 4, 4)
    d_hat = torch.randn(2, 4, 4, 4)
    x_hat = decoder(x_aux, s_hat, d_hat, rate_level=0)
    assert torch.equal(x_hat, x_aux)

