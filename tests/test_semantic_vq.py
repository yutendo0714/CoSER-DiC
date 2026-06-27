import torch

from coserdic.losses import Stage1LossWeights, Stage1SemanticVQLoss
from coserdic.models import SemanticVQAutoEncoder, SemanticVQConfig


def test_semantic_vq_forward_backward_small() -> None:
    cfg = SemanticVQConfig(
        base_channels=16,
        latent_channels=32,
        codebook_size=32,
        num_res_blocks=1,
    )
    model = SemanticVQAutoEncoder(cfg)
    loss_fn = Stage1SemanticVQLoss(
        Stage1LossWeights(l1_sem=1.0, ms_ssim_sem=0.0, lpips_sem=0.0, codebook_usage=0.0)
    )
    x = torch.rand(2, 3, 256, 256)
    out = model(x)
    assert out["x_sem"].shape == x.shape
    assert out["indices"].shape == (2, 8, 8)
    assert out["perplexity"].item() >= 1.0
    assert out["soft_perplexity"].item() >= 1.0
    assert out["assignment_sample_entropy_bits"].item() >= 0.0
    assert out["assignment_avg_entropy_bits"].item() >= 0.0
    assert out["soft_usage_entropy_bits"].item() >= 0.0
    loss = loss_fn(x, out)["total"]
    loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_semantic_vq_quantize_mix_validation() -> None:
    cfg = SemanticVQConfig(
        base_channels=16,
        latent_channels=32,
        codebook_size=32,
        num_res_blocks=1,
    )
    model = SemanticVQAutoEncoder(cfg)
    x = torch.rand(1, 3, 256, 256)
    continuous = model(x, quantize_mix=0.0)
    quantized = model(x, quantize_mix=1.0)
    assert continuous["x_sem"].shape == quantized["x_sem"].shape == x.shape
    try:
        model(x, quantize_mix=1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid quantize_mix should raise ValueError")


def test_soft_st_sends_gradient_to_non_ema_codebook() -> None:
    cfg = SemanticVQConfig(
        base_channels=16,
        latent_channels=32,
        codebook_size=32,
        ema_update=False,
        num_res_blocks=1,
        soft_st=True,
    )
    model = SemanticVQAutoEncoder(cfg)
    x = torch.rand(1, 3, 256, 256)
    out = model(x)
    out["x_sem"].mean().backward()
    assert model.vq.embedding.weight.grad is not None
