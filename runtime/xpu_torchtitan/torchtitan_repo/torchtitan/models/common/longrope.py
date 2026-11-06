"""Non-uniform LongRoPE with an original-RoPE prefix at absolute document positions."""

import math
from dataclasses import dataclass
from typing import Literal

import torch

from torchtitan.models.common.rope import ComplexRoPE


class LongRoPE(ComplexRoPE):
    @dataclass(kw_only=True, slots=True)
    class Config(ComplexRoPE.Config):
        scaling: Literal["longrope"] = "longrope"
        long_factor: tuple[float, ...] = ()
        short_factor: tuple[float, ...] = ()
        target_length: int = 0
        retained_tokens: int = 0

    def _precompute_cache(self):
        cfg = self.config
        factors = cfg.long_factor if cfg.max_seq_len > cfg.original_seq_len else cfg.short_factor
        if len(factors) != cfg.dim // 2 or any(not math.isfinite(v) or v <= 0 for v in factors):
            raise ValueError("Invalid LongRoPE factors")
        inverse = 1.0 / (torch.tensor(factors, dtype=torch.float32)
                         * cfg.theta ** (torch.arange(0, cfg.dim, 2).float() / cfg.dim))
        angles = torch.outer(torch.arange(cfg.max_seq_len, dtype=torch.float32), inverse)
        amplitude = math.sqrt(1 + math.log(cfg.target_length / cfg.original_seq_len)
                              / math.log(cfg.original_seq_len))
        magnitudes = torch.full_like(angles, amplitude)
        if type(cfg.retained_tokens) is not int or not 0 <= cfg.retained_tokens < cfg.original_seq_len:
            raise ValueError("Invalid retained token count")
        count = min(cfg.retained_tokens, cfg.max_seq_len)
        if count:
            original_inverse = 1 / cfg.theta ** (torch.arange(0, cfg.dim, 2).float() / cfg.dim)
            angles[:count] = torch.outer(torch.arange(count).float(), original_inverse)
            magnitudes[:count] = 1
        return torch.polar(magnitudes, angles)
