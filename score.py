"""Score one LongRoPE candidate on allocated XPUs; subprocess lifetime bounds vLLM memory."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

HOME = Path(__file__).resolve().parent
sys.path.insert(0, str(HOME / "runtime"))
from longrope_profile import load, patch_model, save  # noqa: E402


def model_view(source, destination, profile):
    destination.mkdir(exist_ok=True)
    assets = {"generation_config.json", "special_tokens_map.json", "tokenizer.json",
              "tokenizer.model", "tokenizer_config.json", "model.safetensors.index.json"}
    for path in source.iterdir():
        if path.is_file() and (path.name in assets or path.suffix == ".safetensors"):
            link = destination / path.name
            if not link.exists():
                link.symlink_to(path.resolve())
    # Never patch a symlink to the input model's config.
    config = destination / "config.json"
    if config.is_symlink():
        raise ValueError(f"Refusing to overwrite input config symlink: {config}")
    config.write_text((source / "config.json").read_text())
    patch_model(destination, profile)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("profile", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tp", type=int, choices=(1, 4), required=True)
    args = parser.parse_args()
    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    if not torch.xpu.is_available():
        raise RuntimeError("Candidate scoring requires an allocated XPU; no CPU/CUDA fallback")
    profile = load(args.profile, source=args.source)
    data = json.loads(args.data.read_text())
    if data["split"] != "validation" or data["target_length"] != profile["target_length"]:
        raise ValueError("Expected length-matched held-out validation calibration")
    calibration_sha256 = hashlib.sha256(args.data.read_bytes()).hexdigest()
    profile_sha256 = hashlib.sha256(args.profile.read_bytes()).hexdigest()
    partial_path = args.output.parent / "partial.json"
    partial = json.loads(partial_path.read_text()) if partial_path.exists() else {
        "profile_sha256": profile_sha256, "calibration_sha256": calibration_sha256,
        "documents": []}
    if (partial["profile_sha256"] != profile_sha256
            or partial["calibration_sha256"] != calibration_sha256
            or len(partial["documents"]) > len(data["samples"])):
        raise ValueError("Candidate/checkpoint/calibration changed on scoring resume")
    view = args.output.parent / "model"
    model_view(args.source, view, profile)
    engine = LLM(model=str(view), tokenizer=str(view), dtype="bfloat16",
                 tensor_parallel_size=args.tp, max_model_len=profile["target_length"],
                 gpu_memory_utilization=0.8, max_num_seqs=1, max_num_batched_tokens=4096,
                 enforce_eager=True, enable_prefix_caching=False, enable_chunked_prefill=True,
                 worker_cls="longrope_vllm.RetainedXPUWorker",
                 seed=0)
    tokenizer = engine.get_tokenizer()
    prefix = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    if prefix is None:
        raise ValueError("Tokenizer has no BOS/EOS likelihood prefix")
    for row in data["samples"][len(partial["documents"]):]:
        negative_logs = []
        ids = [prefix, *row["tokens"]]
        if len(ids) != profile["target_length"] - 1:
            raise ValueError("Calibration must reserve one prefix and one generated token")
        output = engine.generate([TokensPrompt(prompt_token_ids=ids)],
                                 SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1),
                                 use_tqdm=False)[0]
        if output.prompt_logprobs is None or len(output.prompt_logprobs) != len(ids):
            raise ValueError("Missing or incomplete prompt likelihoods")
        for token, probabilities in zip(ids[1:], output.prompt_logprobs[1:], strict=True):
            value = probabilities[token].logprob
            if not math.isfinite(value) or value > 0:
                raise ValueError("Non-finite/positive candidate log probability")
            negative_logs.append(-value)
        partial["documents"].append({"nll_sum": math.fsum(negative_logs),
                                      "tokens": len(ids) - 1, "document": row["document"]})
        save(partial_path, partial)
    count = sum(doc["tokens"] for doc in partial["documents"])
    save(args.output, {"nll": math.fsum(doc["nll_sum"] for doc in partial["documents"]) / count,
                       "scored_tokens": count, "calibration_sha256": calibration_sha256})


if __name__ == "__main__":
    main()
