import torch

from coserdic.models import CausalTokenPrior, CausalTokenPriorConfig, shifted_causal_inputs


def test_shifted_causal_inputs() -> None:
    targets = torch.tensor([[3, 4, 5], [6, 7, 8]], dtype=torch.long)
    shifted = shifted_causal_inputs(targets, bos_token_id=10)

    assert torch.equal(shifted, torch.tensor([[10, 3, 4], [10, 6, 7]]))


def test_causal_token_prior_forward_shape() -> None:
    cfg = CausalTokenPriorConfig(vocab_size=16, context_length=4, d_model=32, num_layers=1, num_heads=4)
    model = CausalTokenPrior(cfg)
    targets = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)
    inputs = shifted_causal_inputs(targets, bos_token_id=model.bos_token_id)

    logits = model(inputs)

    assert logits.shape == (2, 4, 16)
