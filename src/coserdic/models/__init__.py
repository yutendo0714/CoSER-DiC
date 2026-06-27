"""Model components for CoSER-DiC."""

from .conditioning_adapter import CoSERConditioningAdapter, ConditioningAdapterConfig
from .coserdic import CoSERDiC, CoSEROutput
from .diffusion_decoder import CoSERDiffusionDecoder, DiffusionDecoderConfig
from .residual_detail import (
    ResidualDetailAutoEncoder,
    ResidualDetailConfig,
    ResidualDetailDecoder,
    ResidualDetailEncoder,
    UniformScalarQuantizer,
)
from .semantic_vq import (
    DifferentiableVQ,
    SemanticAuxiliaryDecoder,
    SemanticEncoder,
    SemanticVQAutoEncoder,
    SemanticVQConfig,
)
from .token_prior import CausalTokenPrior, CausalTokenPriorConfig, decoder_schedule_topk_indices, shifted_causal_inputs, topk_from_prefix

__all__ = [
    "CoSERConditioningAdapter",
    "CoSERDiC",
    "CoSERDiffusionDecoder",
    "CoSEROutput",
    "ConditioningAdapterConfig",
    "CausalTokenPrior",
    "CausalTokenPriorConfig",
    "decoder_schedule_topk_indices",
    "DifferentiableVQ",
    "DiffusionDecoderConfig",
    "ResidualDetailAutoEncoder",
    "ResidualDetailConfig",
    "ResidualDetailDecoder",
    "ResidualDetailEncoder",
    "SemanticAuxiliaryDecoder",
    "SemanticEncoder",
    "SemanticVQAutoEncoder",
    "SemanticVQConfig",
    "shifted_causal_inputs",
    "topk_from_prefix",
    "UniformScalarQuantizer",
]
