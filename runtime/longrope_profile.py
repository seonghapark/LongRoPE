"""Validated, source-bound LongRoPE profiles, including custom retained-prefix metadata."""

import hashlib
import json
import math
import os
from pathlib import Path


def save(path, data):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def fingerprint(model):
    model = Path(model).resolve()
    files = [model / "config.json", *sorted(model.glob("*.safetensors"))]
    verified = model / "VERIFIED.json"
    payload = {"model": str(model), "config": files[0].read_text(),
               "weights": [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in files[1:]],
               "verified": verified.read_text() if verified.exists() else None}
    if not payload["weights"]:
        raise FileNotFoundError(f"No safetensors weights: {model}")
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate(profile, *, length=None, dim=None, source=None, searched=False):
    target = profile["target_length"]
    original = profile["original_length"]
    head_dim = profile["head_dim"]
    if (type(target) is not int or type(original) is not int or original < 2
            or target <= original or type(head_dim) is not int or head_dim <= 0
            or head_dim % 2 or not math.isfinite(profile["theta"]) or profile["theta"] <= 1):
        raise ValueError("Invalid LongRoPE geometry")
    retained = profile.get("retained_tokens")
    if (profile.get("rope_type") != "longrope" or type(retained) is not int
            or not 0 <= retained < original):
        raise ValueError("retained_tokens must be an integer in [0, original_length)")
    for key in ("long_factor", "short_factor"):
        values = profile[key]
        if (len(values) != head_dim // 2 or
                any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                    for v in values)):
            raise ValueError(f"{key} must have {head_dim // 2} finite positive entries")
    if length is not None and target != length:
        raise ValueError(f"Profile target {target} != requested {length}")
    if dim is not None and head_dim != dim:
        raise ValueError(f"Profile head_dim {head_dim} != requested {dim}")
    if source is not None and profile["source_fingerprint"] != fingerprint(source):
        raise ValueError(f"Profile was searched against another checkpoint: {source}")
    if searched and (profile.get("status") != "searched"
                     or not math.isfinite(profile.get("validation_nll", float("nan")))):
        raise ValueError("Training/export requires a completed search, not an initialization")
    return profile


def load(path=None, **kwargs):
    path = Path(path or os.environ["LONGROPE_PROFILE"])
    return validate(json.loads(path.read_text()), **kwargs)


def scaling(profile):
    validate(profile)
    factor = profile["target_length"] / profile["original_length"]
    return {"rope_type": "longrope", "factor": factor,
            "original_max_position_embeddings": profile["original_length"],
            "short_factor": profile["short_factor"], "long_factor": profile["long_factor"],
            "attention_factor": math.sqrt(
                1 + math.log(factor) / math.log(profile["original_length"]))}


def patch_model(model, profile):
    model = Path(model)
    cfg = json.loads((model / "config.json").read_text())
    dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    validate(profile, dim=dim)
    if cfg["rope_theta"] != profile["theta"]:
        raise ValueError("Profile theta differs from model")
    cfg.pop("rope_parameters", None)
    cfg["max_position_embeddings"] = profile["target_length"]
    cfg["original_max_position_embeddings"] = profile["original_length"]
    cfg["rope_scaling"] = scaling(profile)
    cfg["longrope_retained_tokens"] = profile["retained_tokens"]
    cfg["longrope_requires_custom_runtime"] = True
    cfg["auto_map"] = {"AutoModelForCausalLM": "modeling_longrope.RetainedLlamaForCausalLM"}
    implementation = Path(__file__).with_name("modeling_longrope.py")
    (model / implementation.name).write_text(implementation.read_text())
    save(model / "config.json", cfg)
    save(model / "LONGROPE_PROFILE.json", profile)
