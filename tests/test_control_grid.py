from __future__ import annotations

import torch

from coserdic.entropy import (
    ComponentCodebookControlCode,
    ComponentMuLawControlCode,
    ComponentUniformControlCode,
    MuLawControlGridCode,
    PrefixTopKControlBasisCode,
    SparseControlBasisCode,
    StaticControlHuffmanCode,
    UniformControlGridCode,
    VectorCodebookControlCode,
    build_control_grid_code,
    dct2_orthonormal,
    idct2_orthonormal,
    project_onto_control_basis,
    reconstruct_from_control_basis,
    zigzag_indices,
)
from scripts.eval_stage4_cod_lite_adapter import (
    grouped_condition_basis_control,
    grouped_condition_dct_control,
    grouped_condition_residual_control,
)


def test_uniform_control_grid_payload_length_and_roundtrip() -> None:
    codec = UniformControlGridCode(bits=4, value_range=0.5)
    values = torch.linspace(-0.5, 0.5, steps=8 * 4 * 4).reshape(8, 4, 4)

    payload = codec.encode_values(values)
    decoded = codec.decode_values(payload, shape=tuple(values.shape))

    assert len(payload) == codec.encoded_num_bytes(tuple(values.shape))
    assert len(payload) == 64
    assert decoded.shape == values.shape
    assert torch.max(torch.abs(decoded - values)) <= (1.0 / 15.0) + 1.0e-6


def test_mulaw_control_grid_roundtrip_spends_levels_near_zero() -> None:
    uniform = UniformControlGridCode(bits=4, value_range=1.0)
    mulaw = MuLawControlGridCode(bits=4, value_range=1.0, mu=16.0)
    values = torch.tensor([-0.05, 0.0, 0.05])

    uniform_decoded = uniform.dequantize(uniform.quantize(values))
    mulaw_decoded = mulaw.dequantize(mulaw.quantize(values))

    assert torch.mean(torch.abs(mulaw_decoded - values)) < torch.mean(torch.abs(uniform_decoded - values))
    assert build_control_grid_code(quantizer="mu_law", bits=4, value_range=1.0, mu=16.0).levels == 16


def test_component_uniform_control_code_uses_per_symbol_ranges() -> None:
    scalar = UniformControlGridCode(bits=4, value_range=1.0)
    component = ComponentUniformControlCode(bits=4, value_ranges=(0.1, 1.0))
    values = torch.tensor([0.05, 0.5])

    scalar_decoded = scalar.dequantize(scalar.quantize(values))
    component_payload = component.encode_values(values)
    component_decoded = component.decode_values(component_payload, shape=(2,))

    assert len(component_payload) == scalar.encoded_num_bytes((2,))
    assert component.encoded_num_bits((2,)) == 8
    assert torch.abs(component_decoded[0] - values[0]) < torch.abs(scalar_decoded[0] - values[0])
    assert ComponentUniformControlCode.from_dict(component.to_dict()).value_ranges == (0.1, 1.0)


def test_component_mulaw_control_code_roundtrip() -> None:
    code = ComponentMuLawControlCode(bits=4, value_ranges=(0.1, 1.0), mu=16.0)
    values = torch.tensor([0.02, -0.4])

    payload = code.encode_values(values)
    decoded = code.decode_values(payload, shape=(2,))

    assert len(payload) == code.encoded_num_bytes((2,))
    assert decoded.shape == values.shape
    assert build_control_grid_code(
        quantizer="mu_law",
        bits=4,
        value_range=1.0,
        value_ranges=[0.1, 1.0],
        mu=16.0,
    ).encoded_num_bytes((2,)) == 1


def test_component_codebook_control_code_uses_fixed_component_centroids() -> None:
    uniform = ComponentUniformControlCode(bits=2, value_ranges=(1.0, 1.0))
    codebook = ComponentCodebookControlCode(
        bits=2,
        codebooks=(
            (-0.10, -0.02, 0.02, 0.10),
            (-1.0, -0.25, 0.25, 1.0),
        ),
    )
    values = torch.tensor([[0.018, -0.23], [0.095, 0.24]])

    uniform_error = torch.mean(torch.abs(uniform.dequantize(uniform.quantize(values)) - values))
    codebook_payload = codebook.encode_values(values[0])
    codebook_decoded = codebook.decode_values(codebook_payload, shape=(2,))
    codebook_error = torch.mean(torch.abs(codebook.dequantize(codebook.quantize(values)) - values))
    selected = codebook.select(torch.tensor([1]))

    assert len(codebook_payload) == uniform.encoded_num_bytes((2,))
    assert codebook.encoded_num_bits((2,)) == 4
    assert codebook_error < uniform_error
    assert torch.allclose(codebook_decoded, torch.tensor([0.02, -0.25]))
    assert selected.component_count == 1
    assert ComponentCodebookControlCode.from_dict(codebook.to_dict()).component_count == 2


def test_vector_codebook_control_code_roundtrip() -> None:
    codec = VectorCodebookControlCode(
        bits=2,
        vectors=(
            (-1.0, -1.0),
            (-0.1, 0.1),
            (0.2, 0.3),
            (1.0, 1.0),
        ),
    )
    values = torch.tensor([[0.18, 0.31], [-0.05, 0.08]])

    codes = codec.quantize(values)
    payload = codec.encode(codes)
    decoded_codes = codec.decode(payload, shape=(2,))
    decoded = codec.dequantize(decoded_codes)

    assert codes.tolist() == [2, 1]
    assert len(payload) == codec.encoded_num_bytes((2,))
    assert decoded.shape == values.shape
    assert torch.allclose(decoded[0], torch.tensor([0.2, 0.3]))
    assert VectorCodebookControlCode.from_dict(codec.to_dict()).vector_dim == 2


def test_grouped_condition_residual_control_counts_payload_bytes() -> None:
    codec = UniformControlGridCode(bits=3, value_range=0.25)
    error = torch.randn(2, 6, 8, 8) * 0.05

    correction, payload_bytes, control_abs = grouped_condition_residual_control(
        error,
        groups=3,
        grid_size=2,
        codec=codec,
        scale=1.0,
    )

    assert correction.shape == error.shape
    assert payload_bytes == [codec.encoded_num_bytes((3, 2, 2)), codec.encoded_num_bytes((3, 2, 2))]
    assert control_abs.shape == (2,)
    assert torch.all(control_abs >= 0)


def test_uniform_control_grid_rejects_invalid_shape_for_byte_count() -> None:
    codec = UniformControlGridCode(bits=4, value_range=0.25)

    try:
        codec.encoded_num_bytes((2, 0, 2))
    except ValueError as exc:
        assert "shape dimensions" in str(exc)
    else:
        raise AssertionError("invalid shape should fail")


def test_static_control_huffman_roundtrip_and_prefix() -> None:
    counts = torch.tensor(
        [
            [10, 2, 1, 1],
            [1, 8, 2, 1],
            [1, 1, 7, 3],
        ],
        dtype=torch.long,
    )
    code = StaticControlHuffmanCode.from_counts(counts, smoothing_count=1)
    values = torch.tensor([0, 1, 2], dtype=torch.long)

    payload = code.encode(values)
    decoded = code.decode(payload, shape=(3,))
    prefix = code.prefix(2)
    prefix_payload = prefix.encode(values[:2])
    prefix_decoded = prefix.decode(prefix_payload, shape=(2,))

    assert torch.equal(decoded, values)
    assert torch.equal(prefix_decoded, values[:2])
    assert StaticControlHuffmanCode.from_dict(code.to_dict()).symbol_shape == (3,)


def test_sparse_control_basis_code_transmits_indices_and_values() -> None:
    value_codec = UniformControlGridCode(bits=4, value_range=1.0)
    code = SparseControlBasisCode(candidate_components=16, selected_components=3, value_codec=value_codec)
    coeffs = torch.tensor([0.1, -2.0, 0.0, 0.5, 1.5, 0.2, -0.7, 0.3, 0.4, -0.6, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0])

    indices, values = code.select(coeffs)
    value_codes = value_codec.quantize(values)
    index_payload, value_payload = code.encode(indices, value_codes)
    compact_payload = code.encode_compact(indices, value_codes)
    decoded_indices, decoded_codes = code.decode(index_payload, value_payload)
    compact_indices, compact_codes = code.decode_compact(compact_payload)
    decoded_values = value_codec.dequantize(decoded_codes)
    sparse = code.scatter_values(decoded_indices, decoded_values)
    index_huffman = StaticControlHuffmanCode.from_counts(torch.ones(3, 16, dtype=torch.long), smoothing_count=1)
    value_huffman = StaticControlHuffmanCode.from_counts(torch.ones(3, value_codec.levels, dtype=torch.long), smoothing_count=1)
    index_payload_h, value_payload_h = code.encode_entropy(
        indices,
        value_codes,
        index_huffman=index_huffman,
        value_huffman=value_huffman,
    )
    decoded_indices_h, decoded_codes_h = code.decode_entropy(
        index_payload_h,
        value_payload_h,
        index_huffman=index_huffman,
        value_huffman=value_huffman,
    )

    assert indices.tolist() == [1, 4, 12]
    assert len(index_payload) + len(value_payload) == code.encoded_num_bytes()
    assert len(compact_payload) == code.encoded_compact_num_bytes()
    assert code.encoded_compact_num_bits() == code.encoded_num_bits()
    assert len(index_payload) == 2
    assert len(value_payload) == 2
    assert torch.equal(decoded_indices, indices)
    assert torch.equal(compact_indices, indices)
    assert torch.equal(compact_codes, value_codes)
    assert torch.equal(decoded_indices_h, indices)
    assert torch.equal(decoded_codes_h, value_codes)
    assert torch.count_nonzero(sparse).item() == 3


def test_prefix_topk_control_basis_code_transmits_prefix_and_sparse_tail() -> None:
    value_codec = UniformControlGridCode(bits=3, value_range=1.0)
    code = PrefixTopKControlBasisCode(
        candidate_components=8,
        prefix_components=2,
        selected_components=2,
        value_codec=value_codec,
    )
    coeffs = torch.tensor([0.2, -0.1, 0.0, 0.9, -0.8, 0.1, 0.4, -0.2])

    prefix_values, local_indices, global_indices, selected_values = code.select(coeffs)
    prefix_codes = value_codec.quantize(prefix_values)
    selected_codes = value_codec.quantize(selected_values)
    payload = code.encode_compact(prefix_codes, local_indices, selected_codes)
    decoded_prefix_codes, decoded_local_indices, decoded_selected_codes = code.decode_compact(payload)
    decoded_prefix = value_codec.dequantize(decoded_prefix_codes)
    decoded_selected = value_codec.dequantize(decoded_selected_codes)
    sparse = code.scatter_values(decoded_prefix, decoded_local_indices, decoded_selected)

    assert global_indices.tolist() == [3, 4]
    assert local_indices.tolist() == [1, 2]
    assert len(payload) == code.encoded_compact_num_bytes()
    assert code.encoded_compact_num_bits() == 2 * 3 + 2 * (3 + 3)
    assert torch.equal(decoded_local_indices, local_indices)
    assert torch.equal(decoded_selected_codes, selected_codes)
    assert torch.count_nonzero(sparse).item() == 4


def test_sparse_control_basis_code_accepts_component_codebook_values() -> None:
    value_codec = ComponentCodebookControlCode(
        bits=2,
        codebooks=(
            (-0.2, -0.1, 0.1, 0.2),
            (-1.2, -0.6, 0.6, 1.2),
            (-0.2, -0.1, 0.1, 0.2),
            (-0.3, -0.15, 0.15, 0.3),
        ),
    )
    code = SparseControlBasisCode(candidate_components=4, selected_components=2, value_codec=value_codec)
    coeffs = torch.tensor([0.05, -1.0, 0.0, 0.15])

    indices, values = code.select(coeffs)
    selected_codec = value_codec.select(indices)
    value_codes = selected_codec.quantize(values)
    selected_code = SparseControlBasisCode(candidate_components=4, selected_components=2, value_codec=selected_codec)
    index_payload, value_payload = selected_code.encode(indices, value_codes)
    decoded_indices, decoded_codes = selected_code.decode(index_payload, value_payload)
    decoded_values = selected_codec.dequantize(decoded_codes)

    assert indices.tolist() == [1, 3]
    assert len(index_payload) + len(value_payload) == selected_code.encoded_num_bytes()
    assert torch.equal(decoded_indices, indices)
    assert torch.allclose(decoded_values, torch.tensor([-1.2, 0.15]))


def test_dct2_orthonormal_roundtrip_and_zigzag() -> None:
    x = torch.randn(2, 3, 4, 4)
    coeffs = dct2_orthonormal(x)
    recon = idct2_orthonormal(coeffs)

    assert torch.allclose(recon, x.float(), atol=1.0e-5)
    assert zigzag_indices(4, 6) == [(0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (2, 0)]


def test_grouped_condition_dct_control_uses_selected_coeff_payload_bytes() -> None:
    codec = UniformControlGridCode(bits=4, value_range=0.5)
    error = torch.randn(2, 8, 8, 8) * 0.05

    correction, payload_bytes, coeff_abs = grouped_condition_dct_control(
        error,
        groups=4,
        grid_size=4,
        coeffs_per_group=3,
        codec=codec,
        scale=0.5,
    )

    assert correction.shape == error.shape
    assert payload_bytes == [codec.encoded_num_bytes((4, 3)), codec.encoded_num_bytes((4, 3))]
    assert payload_bytes == [6, 6]
    assert coeff_abs.shape == (2,)


def test_control_basis_projection_and_reconstruction() -> None:
    basis = torch.zeros(2, 2, 2, 2)
    basis[0, 0, 0, 0] = 1.0
    basis[1, 1, 1, 1] = 1.0
    mean = torch.full((2, 2, 2), 0.25)
    grid = mean.unsqueeze(0).repeat(3, 1, 1, 1)
    grid[:, 0, 0, 0] += torch.tensor([1.0, 2.0, -1.0])
    grid[:, 1, 1, 1] += torch.tensor([0.5, -0.5, 1.5])

    coeffs = project_onto_control_basis(grid, basis, mean, components=2)
    recon = reconstruct_from_control_basis(coeffs, basis, mean)

    assert coeffs.shape == (3, 2)
    assert torch.allclose(recon, grid)


def test_grouped_condition_basis_control_counts_coeff_payload_bytes() -> None:
    codec = UniformControlGridCode(bits=4, value_range=0.5)
    basis = torch.zeros(3, 4, 2, 2)
    basis[0, :, 0, 0] = 0.5
    basis[1, :, 0, 1] = 0.5
    basis[2, :, 1, 0] = 0.5
    basis_payload = {"basis": basis, "mean": torch.zeros(4, 2, 2), "groups": 4, "grid_size": 2}
    error = torch.randn(2, 8, 8, 8) * 0.05

    correction, payload_bytes, coeff_abs = grouped_condition_basis_control(
        error,
        basis_payload=basis_payload,
        components=3,
        candidate_components=3,
        selection="prefix",
        codec=codec,
        scale=1.0,
    )

    assert correction.shape == error.shape
    assert payload_bytes == [codec.encoded_num_bytes((3,)), codec.encoded_num_bytes((3,))]
    assert payload_bytes == [2, 2]
    assert coeff_abs.shape == (2,)


def test_grouped_condition_basis_control_accepts_component_ranges() -> None:
    codec = ComponentUniformControlCode(bits=4, value_ranges=(0.05, 0.5, 1.0))
    basis = torch.zeros(3, 4, 2, 2)
    basis[0, :, 0, 0] = 0.5
    basis[1, :, 0, 1] = 0.5
    basis[2, :, 1, 0] = 0.5
    basis_payload = {"basis": basis, "mean": torch.zeros(4, 2, 2), "groups": 4, "grid_size": 2}
    error = torch.randn(2, 8, 8, 8) * 0.05

    correction, payload_bytes, coeff_abs = grouped_condition_basis_control(
        error,
        basis_payload=basis_payload,
        components=3,
        candidate_components=3,
        selection="prefix",
        codec=codec,
        scale=1.0,
    )

    assert correction.shape == error.shape
    assert payload_bytes == [codec.encoded_num_bytes((3,)), codec.encoded_num_bytes((3,))]
    assert coeff_abs.shape == (2,)


def test_grouped_condition_basis_control_supports_huffman_payload_bytes() -> None:
    codec = UniformControlGridCode(bits=2, value_range=0.5)
    basis = torch.zeros(3, 4, 2, 2)
    basis[0, :, 0, 0] = 0.5
    basis[1, :, 0, 1] = 0.5
    basis[2, :, 1, 0] = 0.5
    basis_payload = {"basis": basis, "mean": torch.zeros(4, 2, 2), "groups": 4, "grid_size": 2}
    huffman = StaticControlHuffmanCode.from_counts(torch.ones(3, codec.levels, dtype=torch.long), smoothing_count=1)
    error = torch.randn(2, 8, 8, 8) * 0.05

    correction, payload_bytes, coeff_abs = grouped_condition_basis_control(
        error,
        basis_payload=basis_payload,
        components=3,
        candidate_components=3,
        selection="prefix",
        codec=codec,
        huffman=huffman,
        scale=1.0,
    )

    assert correction.shape == error.shape
    assert payload_bytes == [1, 1]
    assert coeff_abs.shape == (2,)


def test_grouped_condition_basis_control_supports_vector_codebook_payload_bytes() -> None:
    basis = torch.zeros(2, 4, 2, 2)
    basis[0, :, 0, 0] = 0.5
    basis[1, :, 0, 1] = 0.5
    basis_payload = {"basis": basis, "mean": torch.zeros(4, 2, 2), "groups": 4, "grid_size": 2}
    error = torch.zeros(1, 4, 8, 8)
    error[:, :, :4, :4] = 0.2
    error[:, :, :4, 4:] = -0.1
    codec = VectorCodebookControlCode(
        bits=2,
        vectors=(
            (0.0, 0.0),
            (0.4, -0.2),
            (-0.4, 0.2),
            (0.8, -0.4),
        ),
    )

    correction, payload_bytes, coeff_abs = grouped_condition_basis_control(
        error,
        basis_payload=basis_payload,
        components=2,
        candidate_components=2,
        selection="vector",
        codec=codec,
        scale=1.0,
    )

    assert correction.shape == error.shape
    assert payload_bytes == [codec.encoded_num_bytes((1,))]
    assert coeff_abs.shape == (1,)
    assert torch.all(coeff_abs > 0)


def test_grouped_condition_basis_control_supports_sparse_topk_payload_bytes() -> None:
    codec = UniformControlGridCode(bits=4, value_range=0.5)
    basis = torch.zeros(8, 4, 2, 2)
    for index in range(8):
        basis[index, index % 4, (index // 4) % 2, index % 2] = 1.0
    basis_payload = {"basis": basis, "mean": torch.zeros(4, 2, 2), "groups": 4, "grid_size": 2}
    error = torch.randn(2, 8, 8, 8) * 0.05

    correction, payload_bytes, coeff_abs = grouped_condition_basis_control(
        error,
        basis_payload=basis_payload,
        components=2,
        candidate_components=8,
        selection="topk",
        codec=codec,
        scale=1.0,
    )

    assert correction.shape == error.shape
    assert payload_bytes == [2, 2]
    assert coeff_abs.shape == (2,)


def test_grouped_condition_basis_control_compacts_small_sparse_topk_payload() -> None:
    codec = UniformControlGridCode(bits=2, value_range=0.5)
    basis = torch.zeros(3, 3, 2, 2)
    for index in range(3):
        basis[index, index, 0, index % 2] = 1.0
    basis_payload = {"basis": basis, "mean": torch.zeros(3, 2, 2), "groups": 3, "grid_size": 2}
    error = torch.randn(1, 3, 8, 8) * 0.05

    correction, payload_bytes, coeff_abs = grouped_condition_basis_control(
        error,
        basis_payload=basis_payload,
        components=1,
        candidate_components=3,
        selection="topk",
        codec=codec,
        scale=1.0,
    )

    assert correction.shape == error.shape
    assert payload_bytes == [1]
    assert coeff_abs.shape == (1,)


def test_grouped_condition_basis_control_supports_prefix_topk_payload_bytes() -> None:
    codec = UniformControlGridCode(bits=4, value_range=0.5)
    basis = torch.zeros(8, 4, 2, 2)
    for index in range(8):
        basis[index, index % 4, (index // 4) % 2, index % 2] = 1.0
    basis_payload = {"basis": basis, "mean": torch.zeros(4, 2, 2), "groups": 4, "grid_size": 2}
    error = torch.randn(2, 8, 8, 8) * 0.05

    correction, payload_bytes, coeff_abs = grouped_condition_basis_control(
        error,
        basis_payload=basis_payload,
        components=2,
        candidate_components=8,
        prefix_components=2,
        selection="prefix_topk",
        codec=codec,
        scale=1.0,
    )

    assert correction.shape == error.shape
    assert payload_bytes == [3, 3]
    assert coeff_abs.shape == (2,)


def test_grouped_condition_basis_control_supports_sparse_topk_component_ranges() -> None:
    codec = ComponentUniformControlCode(bits=4, value_ranges=(0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.2, 1.5))
    basis = torch.zeros(8, 4, 2, 2)
    for index in range(8):
        basis[index, index % 4, (index // 4) % 2, index % 2] = 1.0
    basis_payload = {"basis": basis, "mean": torch.zeros(4, 2, 2), "groups": 4, "grid_size": 2}
    error = torch.randn(2, 8, 8, 8) * 0.05

    correction, payload_bytes, coeff_abs = grouped_condition_basis_control(
        error,
        basis_payload=basis_payload,
        components=2,
        candidate_components=8,
        selection="topk",
        codec=codec,
        scale=1.0,
    )

    assert correction.shape == error.shape
    assert payload_bytes == [2, 2]
    assert coeff_abs.shape == (2,)


def test_grouped_condition_basis_control_supports_sparse_topk_huffman_payload_bytes() -> None:
    codec = UniformControlGridCode(bits=2, value_range=0.5)
    basis = torch.zeros(8, 4, 2, 2)
    for index in range(8):
        basis[index, index % 4, (index // 4) % 2, index % 2] = 1.0
    basis_payload = {"basis": basis, "mean": torch.zeros(4, 2, 2), "groups": 4, "grid_size": 2}
    index_huffman = StaticControlHuffmanCode.from_counts(torch.ones(2, 8, dtype=torch.long), smoothing_count=1)
    value_huffman = StaticControlHuffmanCode.from_counts(torch.ones(2, codec.levels, dtype=torch.long), smoothing_count=1)
    error = torch.randn(2, 8, 8, 8) * 0.05

    correction, payload_bytes, coeff_abs = grouped_condition_basis_control(
        error,
        basis_payload=basis_payload,
        components=2,
        candidate_components=8,
        selection="topk",
        codec=codec,
        sparse_huffman=(index_huffman, value_huffman),
        scale=1.0,
    )

    assert correction.shape == error.shape
    assert payload_bytes == [2, 2]
    assert coeff_abs.shape == (2,)
