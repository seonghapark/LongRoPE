"""Explicit worker hook so spawned XPU workers cannot lose retained-prefix RoPE."""

import torch
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers import rotary_embedding
from vllm.model_executor.layers.rotary_embedding.phi3_long_rope_scaled_rope import (
    Phi3LongRoPEScaledRotaryEmbedding,
)
from vllm.v1.worker.xpu_worker import XPUWorker


class RetainedLongRoPE(Phi3LongRoPEScaledRotaryEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = get_current_vllm_config().model_config.hf_config
        if not getattr(config, "longrope_requires_custom_runtime", False):
            return
        retained = config.longrope_retained_tokens
        if type(retained) is not int or not 0 <= retained < self.original_max_position_embeddings:
            raise ValueError("Invalid retained-token count in vLLM model config")
        inverse = 1 / self.base ** (
            torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim)
        angles = torch.arange(retained).float()[:, None] * inverse[None, :]
        original = torch.cat((angles.cos(), angles.sin()), dim=-1)
        cache = self.long_short_cos_sin_cache
        for offset in (0, self.original_max_position_embeddings):
            cache[offset:offset + retained] = original.to(cache)


def install():
    rotary_embedding.Phi3LongRoPEScaledRotaryEmbedding = RetainedLongRoPE


class RetainedXPUWorker(XPUWorker):
    def __init__(self, *args, **kwargs):
        install()
        super().__init__(*args, **kwargs)
