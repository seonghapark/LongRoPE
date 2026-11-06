"""CPU-only checks using the actual evaluation venv's vLLM LongRoPE implementation."""

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class VLLMLongRoPETests(unittest.TestCase):
    def test_retained_prefix_through_actual_factory(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent / "runtime"))
        import torch
        import longrope_vllm
        from vllm.model_executor.layers import rotary_embedding

        base = longrope_vllm.Phi3LongRoPEScaledRotaryEmbedding
        for theta in (50000, 500000):
            for cap in (4096, 131072):
                hf = SimpleNamespace(longrope_requires_custom_runtime=True,
                                     longrope_retained_tokens=64)
                config = SimpleNamespace(model_config=SimpleNamespace(
                    max_model_len=cap, hf_config=hf))
                with patch.dict(base.__init__.__globals__,
                                {"get_current_vllm_config": lambda: config}), \
                        patch.object(longrope_vllm, "get_current_vllm_config", return_value=config), \
                        patch.object(rotary_embedding, "Phi3LongRoPEScaledRotaryEmbedding",
                                     longrope_vllm.RetainedLongRoPE):
                    rotary_embedding._ROPE_DICT.clear()
                    rope = rotary_embedding.get_rope(
                        128, 131072, dtype=torch.float32,
                        rope_parameters={"rope_type": "longrope", "rope_theta": theta,
                                         "original_max_position_embeddings": 8192,
                                         "short_factor": [1.0] * 64, "long_factor": [16.0] * 64})
                self.assertIsInstance(rope, longrope_vllm.RetainedLongRoPE)
                positions = torch.tensor([0, 1, 63, 64, cap - 1])
                inverse = 1 / theta ** (torch.arange(0, 128, 2).float() / 128)
                original = positions[:, None] * inverse[None, :]
                scaled = original / (16 if cap > 8192 else 1)
                angles = torch.where((positions < 64)[:, None], original, scaled)
                magnitude = torch.where(
                    (positions < 64)[:, None], 1.0, math.sqrt(1 + math.log(16) / math.log(8192)))
                expected = torch.cat((angles.cos(), angles.sin()), dim=-1) * magnitude
                offset = 8192 if cap > 8192 else 0
                torch.testing.assert_close(rope.long_short_cos_sin_cache[positions + offset],
                                           expected, atol=1e-6, rtol=1e-6)
                query = torch.randn(len(positions), 128)
                key = torch.randn(len(positions), 128)
                q, k = rope(positions, query, key)
                self.assertTrue(torch.isfinite(q).all() and torch.isfinite(k).all())
        rotary_embedding._ROPE_DICT.clear()

    def test_long_and_short_cache_frequency_amplitude_and_selection(self):
        import torch
        from vllm.model_executor.layers.rotary_embedding import get_rope

        cls = get_rope.__globals__["Phi3LongRoPEScaledRotaryEmbedding"]
        for theta in (50000, 500000):
            for target in (16384, 32768, 65536, 131072):
                long = [1 + (target / 8192 - 1) * i / 63 for i in range(64)]
                amplitude = math.sqrt(1 + math.log(target / 8192) / math.log(8192))
                for serving_cap in (4096, target):
                    config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=serving_cap))
                    with patch.dict(cls.__init__.__globals__,
                                    {"get_current_vllm_config": lambda: config}):
                        rope = cls(128, 128, target, 8192, theta, True, torch.float32,
                                   [1.0] * 64, long)
                    self.assertEqual(rope.use_long_rope, serving_cap > 8192)
                    positions = torch.tensor([0, 1, serving_cap - 1])
                    factors = long if rope.use_long_rope else [1.0] * 64
                    inverse = 1 / (torch.tensor(factors) * theta ** (torch.arange(0, 128, 2) / 128))
                    angles = positions[:, None] * inverse[None, :]
                    expected = torch.cat((angles.cos(), angles.sin()), dim=-1) * amplitude
                    offset = 8192 if rope.use_long_rope else 0
                    torch.testing.assert_close(
                        rope.long_short_cos_sin_cache[positions + offset],
                        expected, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
