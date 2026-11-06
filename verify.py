"""Compare exported metadata and actual TorchTitan LongRoPE cache against HF."""

import argparse
import json
import sys
from pathlib import Path

HOME = Path(__file__).resolve().parent
sys.path[:0] = [str(HOME / "runtime"),
               str(HOME / "runtime/xpu_torchtitan/torchtitan_repo")]
from longrope_profile import load, scaling  # noqa: E402


def verify(model):
    import torch
    from transformers import AutoConfig
    from transformers.modeling_rope_utils import _compute_longrope_parameters
    from torchtitan.models.common.longrope import LongRoPE

    profile = load(model / "LONGROPE_PROFILE.json", searched=True)
    raw = json.loads((model / "config.json").read_text())
    if (raw["rope_scaling"] != scaling(profile)
            or raw["max_position_embeddings"] != profile["target_length"]
            or raw["rope_theta"] != profile["theta"]
            or raw.get("longrope_retained_tokens") != profile["retained_tokens"]
            or raw.get("longrope_requires_custom_runtime") is not True):
        raise ValueError("Export config disagrees with searched LongRoPE profile")
    cfg = AutoConfig.from_pretrained(str(model), local_files_only=True)
    for length in (4096, profile["target_length"]):
        rope = LongRoPE(LongRoPE.Config(
            dim=profile["head_dim"], max_seq_len=length, theta=profile["theta"],
            original_seq_len=profile["original_length"],
            target_length=profile["target_length"],
            retained_tokens=profile["retained_tokens"],
            short_factor=tuple(profile["short_factor"]),
            long_factor=tuple(profile["long_factor"])))
        inverse, amplitude = _compute_longrope_parameters(cfg, torch.device("cpu"), seq_len=length)
        positions = torch.tensor([0, 1, length // 2, length - 1])
        angles = positions.float()[:, None] * inverse[None, :]
        expected = torch.polar(torch.full_like(angles, amplitude), angles)
        retained = positions < profile["retained_tokens"]
        original_inverse = 1 / profile["theta"] ** (
            torch.arange(0, profile["head_dim"], 2).float() / profile["head_dim"])
        original_angles = positions.float()[:, None] * original_inverse[None, :]
        expected[retained] = torch.polar(torch.ones_like(original_angles[retained]),
                                         original_angles[retained])
        torch.testing.assert_close(rope.cache[positions], expected, atol=1e-6, rtol=1e-6)
        from modeling_longrope import RetainedRotaryEmbedding

        hf_rope = RetainedRotaryEmbedding(cfg, serving_length=length)
        cos, sin = hf_rope(torch.zeros(1, dtype=torch.float32), positions[None, :])
        torch.testing.assert_close(cos[0, :, :profile["head_dim"] // 2], expected.real)
        torch.testing.assert_close(sin[0, :, :profile["head_dim"] // 2], expected.imag)
    print("ALL MATCH: TorchTitan/custom HF LongRoPE, including retained original prefix")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    verify(parser.parse_args().model)
