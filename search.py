"""Resumable evolutionary search on PG19 validation, never on the test split."""

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HOME = Path(__file__).resolve().parent
RETAINED_CHOICES = (0, 1, 4, 16, 64, 256)
sys.path.insert(0, str(HOME / "runtime"))
import post50  # noqa: E402
from longrope_profile import fingerprint, save, validate  # noqa: E402


def initial_profile(source, length):
    cfg = json.loads((source / "config.json").read_text())
    dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    previous = cfg.get("rope_scaling", {})
    factors = ([v * length / cfg["max_position_embeddings"] for v in previous["long_factor"]]
               if previous.get("rope_type") == "longrope" else [length / 8192] * (dim // 2))
    return {"rope_type": "longrope", "original_length": 8192, "target_length": length,
            "head_dim": dim, "theta": cfg["rope_theta"], "retained_tokens": 0,
            "short_factor": [1.0] * (dim // 2),
            "long_factor": factors,
            "source": str(source), "source_fingerprint": fingerprint(source),
            "status": "candidate"}


def population(template, generation, size, seed, parents=()):
    rng = random.Random(seed + generation)
    maximum = template["target_length"] / template["original_length"]
    vectors = [p["long_factor"][:] for p in parents[:2]]
    if not vectors:
        vectors = [template["long_factor"][:], [1.0] * (template["head_dim"] // 2)]
    while len(vectors) < size:
        if not parents:
            vector = sorted(rng.uniform(1, maximum) for _ in template["long_factor"])
        else:
            a, b = rng.choices(parents, k=2)
            vector = sorted(max(1, min(maximum, (
                a["long_factor"][i] if rng.random() < 0.5 else b["long_factor"][i])
                * math.exp(rng.gauss(0, 0.15))))
                for i in range(len(template["long_factor"])))
        vectors.append(vector)
    retained = ([p["retained_tokens"] for p in parents[:2]] if parents else [0, 0])
    while len(retained) < len(vectors):
        retained.append(RETAINED_CHOICES[(len(retained) - 2) % len(RETAINED_CHOICES)]
                        if not parents else rng.choice(RETAINED_CHOICES))
    candidates = [dict(template, long_factor=v, retained_tokens=retained[i],
                       generation=generation, candidate=i)
                  for i, v in enumerate(vectors)]
    for candidate, parent in zip(candidates, parents[:2]):
        candidate["validation_nll"] = parent["validation_nll"]
        candidate["score_file"] = parent["score_file"]
    return candidates


def calibration(source, length, output, samples):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    files = sorted((post50.BASE / "assets/hf/datasets/pg19/data").glob("validation-*.parquet"))
    if not files:
        raise FileNotFoundError("PG19 validation parquet is required; test is not a fallback")
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True)
    rows = []
    dataset = load_dataset("parquet", data_files={"validation": [str(p) for p in files]},
                           split="validation")
    for index in random.Random(1234).sample(range(len(dataset)), len(dataset)):
        tokens = tokenizer.encode(dataset[index]["text"], add_special_tokens=False)
        if len(tokens) >= length - 2:
            rows.append({"document": index, "tokens": tokens[:length - 2]})
        if len(rows) == samples:
            break
    if len(rows) != samples:
        raise ValueError(f"Only {len(rows)} validation documents are long enough for {length}")
    save(output, {"split": "validation", "source_files": [str(p) for p in files],
                  "tokenizer_sha256": hashlib.sha256((source / "tokenizer.json").read_bytes()).hexdigest(),
                  "samples": rows, "target_length": length})


def run(args):
    deadline = time.monotonic() + args.budget_seconds
    root = args.root.resolve()
    root.relative_to(HOME / "experiments")
    os.environ.update(HF_HOME=str(root / "cache/huggingface"),
                      HF_DATASETS_CACHE=str(root / "cache/datasets"))
    stage = post50.STAGES[args.stage]
    source = post50.source(root, stage)
    destination = root / "profiles" / f"{post50.tag(stage)}.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite an accepted profile: {destination}")
    directory = root / "search" / post50.tag(stage)
    directory.mkdir(exist_ok=True)
    template = initial_profile(source, stage[2])
    data = directory / "calibration.json"
    if not data.exists():
        calibration(source, stage[2], data, args.samples)
    corpus = json.loads(data.read_text())
    if (corpus["split"] != "validation" or corpus["target_length"] != stage[2]
            or len(corpus["samples"]) != args.samples
            or any(len(row["tokens"]) != stage[2] - 2 for row in corpus["samples"])
            or corpus["tokenizer_sha256"] !=
            hashlib.sha256((source / "tokenizer.json").read_bytes()).hexdigest()):
        raise ValueError("Calibration split/length/sample count/tokenizer changed")
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    settings = {"population": args.population, "generations": args.generations,
                "retained_choices": list(RETAINED_CHOICES),
                "seed": args.seed, "samples": args.samples,
                "calibration_sha256": digest, "source_fingerprint": template["source_fingerprint"]}
    state_path = directory / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "settings": settings, "generation": 0, "parents": [], "candidates": []}
    if state["settings"] != settings:
        raise ValueError("Search settings/source/calibration changed on resume")
    while state["generation"] < args.generations:
        if not state["candidates"]:
            state["candidates"] = population(template, state["generation"], args.population,
                                             args.seed, state["parents"])
            save(state_path, state)
        for candidate in state["candidates"]:
            if "validation_nll" in candidate:
                continue
            remaining = deadline - time.monotonic() - 30
            if remaining < 120:
                return 124
            path = directory / f"g{candidate['generation']}_c{candidate['candidate']}"
            path.mkdir(exist_ok=True)
            save(path / "profile.json", candidate)
            result = path / "score.json"
            command = [sys.executable, str(HOME / "score.py"), str(source),
                       str(path / "profile.json"), str(data), str(result),
                       "--tp", "1" if stage[0] == "2b" else "4"]
            env = dict(os.environ, ZE_FLAT_DEVICE_HIERARCHY="FLAT",
                       ZE_AFFINITY_MASK="0" if stage[0] == "2b" else "0,1,2,3",
                       TOKENIZERS_PARALLELISM="false",
                       VLLM_CACHE_ROOT=str(root / "cache/vllm_search"),
                       HF_HOME=str(root / "cache/huggingface"),
                       HF_DATASETS_CACHE=str(root / "cache/datasets"))
            if not result.exists():
                partial = path / "partial.json"
                before = partial.stat().st_mtime_ns if partial.exists() else None
                with (path / "score.log").open("a") as log:
                    try:
                        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       check=True, timeout=remaining)
                    except subprocess.TimeoutExpired:
                        after = partial.stat().st_mtime_ns if partial.exists() else None
                        state["stalled"] = state.get("stalled", 0) + 1 if after == before else 0
                        save(state_path, state)
                        if state["stalled"] >= 3:
                            raise RuntimeError("Three search allocations without document progress")
                        return 124
            score = json.loads(result.read_text())
            nll = score["nll"]
            if (not math.isfinite(nll) or nll < 0
                    or score["scored_tokens"] != args.samples * (stage[2] - 2)
                    or score["calibration_sha256"] != digest):
                raise ValueError(f"Invalid candidate score: {result}")
            candidate["validation_nll"] = nll
            candidate["score_file"] = str(result)
            state["stalled"] = 0
            save(state_path, state)
        state["parents"] = sorted(state["candidates"], key=lambda p: p["validation_nll"])[:2]
        state["generation"] += 1
        state["candidates"] = []
        save(state_path, state)
    winner = dict(state["parents"][0], status="searched", search_settings=settings,
                  calibration=str(data), method="LongRoPE joint dimension/retained-prefix evolution")
    validate(winner, length=stage[2], source=source, searched=True)
    save(destination, winner)
    print(f"Accepted LongRoPE profile: {destination}", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HOME / "experiments/prolong128k")
    parser.add_argument("--stage", type=int, required=True, choices=range(len(post50.STAGES)))
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--budget-seconds", type=float, default=2700)
    args = parser.parse_args()
    if args.population < 3 or args.generations < 1 or args.samples < 1 or args.budget_seconds <= 0:
        parser.error("population>=3, generations>=1, samples>=1, budget-seconds>0 required")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
