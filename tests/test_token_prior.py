import torch

from coserdic.models import (
    CausalTokenPrior,
    CausalTokenPriorConfig,
    decoder_prefix_topk_indices,
    decoder_schedule_topk_indices,
    shifted_causal_inputs,
    topk_from_prefix,
)


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


def test_decoder_schedule_topk_indices_matches_prefix_loop() -> None:
    torch.manual_seed(0)
    cfg = CausalTokenPriorConfig(vocab_size=16, context_length=4, d_model=32, num_layers=1, num_heads=4)
    model = CausalTokenPrior(cfg)
    model.eval()
    indices = torch.tensor([[1, 5], [7, 2]], dtype=torch.long)

    scheduled = decoder_schedule_topk_indices(model, indices, topk=3, device=torch.device("cpu"))

    prefix: list[int] = []
    rows: list[list[int]] = []
    for token in indices.reshape(-1).tolist():
        rows.append(topk_from_prefix(model, prefix, topk=3, device=torch.device("cpu")))
        prefix.append(int(token))
    expected = torch.tensor(rows, dtype=torch.long).reshape(2, 2, 3)
    assert torch.equal(scheduled, expected)


def test_decoder_prefix_topk_indices_matches_prefix_loop() -> None:
    torch.manual_seed(1)
    cfg = CausalTokenPriorConfig(vocab_size=32, context_length=6, d_model=32, num_layers=1, num_heads=4)
    model = CausalTokenPrior(cfg)
    model.eval()
    indices = torch.tensor([[1, 5, 7], [2, 9, 4]], dtype=torch.long)

    scheduled = decoder_prefix_topk_indices(model, indices, topk=5, device=torch.device("cpu"))

    prefix: list[int] = []
    rows: list[list[int]] = []
    for token in indices.reshape(-1).tolist():
        rows.append(topk_from_prefix(model, prefix, topk=5, device=torch.device("cpu")))
        prefix.append(int(token))
    expected = torch.tensor(rows, dtype=torch.long).reshape(2, 3, 5)
    assert torch.equal(scheduled, expected)


def test_decoder_prefix_topk_indices_resets_prefix_per_batch_item() -> None:
    torch.manual_seed(2)
    cfg = CausalTokenPriorConfig(vocab_size=32, context_length=4, d_model=32, num_layers=1, num_heads=4)
    model = CausalTokenPrior(cfg)
    model.eval()
    indices = torch.tensor([[[1, 5], [7, 2]], [[3, 4], [6, 8]]], dtype=torch.long)

    scheduled = decoder_prefix_topk_indices(model, indices, topk=3, device=torch.device("cpu"))

    expected_rows = []
    for image in indices:
        prefix: list[int] = []
        rows: list[list[int]] = []
        for token in image.reshape(-1).tolist():
            rows.append(topk_from_prefix(model, prefix, topk=3, device=torch.device("cpu")))
            prefix.append(int(token))
        expected_rows.append(torch.tensor(rows, dtype=torch.long).reshape(2, 2, 3))
    expected = torch.stack(expected_rows, dim=0)
    assert torch.equal(scheduled, expected)
