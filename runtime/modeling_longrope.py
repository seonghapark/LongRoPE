"""Position-dependent LongRoPE; the retained prefix uses original RoPE and unit amplitude."""

import torch
from transformers.models.llama.modeling_llama import LlamaForCausalLM


class RetainedRotaryEmbedding(torch.nn.Module):
    def __init__(self, config, serving_length=None):
        super().__init__()
        parameters = config.rope_parameters
        self.dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.theta = parameters.get("rope_theta", getattr(config, "rope_theta", 10000))
        self.original = parameters["original_max_position_embeddings"]
        self.target = config.max_position_embeddings
        self.retained = config.longrope_retained_tokens
        self.serving_length = serving_length
        self.register_buffer("exponents", torch.arange(0, self.dim, 2).float() / self.dim,
                             persistent=False)
        self.register_buffer("long_factors", torch.tensor(parameters["long_factor"]).float(),
                             persistent=False)
        self.register_buffer("short_factors", torch.tensor(parameters["short_factor"]).float(),
                             persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        import math

        length = self.serving_length
        if length is None:
            length = int(position_ids.max()) + 1
        factors = self.long_factors if length > self.original else self.short_factors
        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            base = self.theta ** self.exponents.to(x.device)
            positions = position_ids.float().unsqueeze(-1)
            original_angles = positions * (1 / base)
            angles = positions * (1 / (factors.to(x.device) * base))
            keep = (position_ids < self.retained).unsqueeze(-1)
            angles = torch.where(keep, original_angles, angles)
            amplitude = math.sqrt(1 + math.log(self.target / self.original) / math.log(self.original))
            magnitude = torch.where(keep, 1.0, amplitude)
            angles = torch.cat((angles, angles), dim=-1)
            return (angles.cos() * magnitude).to(x.dtype), (angles.sin() * magnitude).to(x.dtype)


class RetainedLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model.rotary_emb = RetainedRotaryEmbedding(
            config, serving_length=config.max_position_embeddings)
