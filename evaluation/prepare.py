"""Prepare standard/long, MMLU-5shot and long-PPL plans for verified LongRoPE models."""

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

HOME = Path(__file__).resolve().parents[1]
BASE = HOME.parent
sys.path.insert(0, str(HOME / "runtime"))
import post50  # noqa: E402
from longrope_profile import load, save  # noqa: E402
from eval_posttrained_2b import build_shards  # noqa: E402


def prepare(root):
    root = root.resolve()
    root.relative_to(HOME / "experiments")
    os.environ.update(HF_HOME=str(root / "cache/huggingface"),
                      HF_DATASETS_CACHE=str(root / "cache/datasets"))
    # Refuse partial/fake milestone sets; preparation never submits any job.
    for stage in post50.STAGES:
        for step in post50.MILESTONES:
            path = post50.model(root, stage, step)
            if not (path / "VERIFIED.json").is_file():
                raise FileNotFoundError(f"Required verified LongRoPE milestone missing: {path}")
            load(path / "LONGROPE_PROFILE.json", length=stage[2], searched=True)
    output = root / "evaluation/sweep"
    output.mkdir(exist_ok=False)
    jobs = []
    for size in ("2b", "20b"):
        original = post50.source(root, (size, "independent", 131072))
        tasks = output / f"{size}_corpora"
        tasks.mkdir()
        from corpora import prepare_corpora

        stats = prepare_corpora(tasks, original, root / "data/prolong")
        save(tasks / "corpora.json", stats)
        for mode, step in [("baseline", 0)] + [
                ("direct", s) for s in post50.MILESTONES]:
            series = output / f"{size}_{mode}_{step}"
            series.mkdir()
            selection = series / "selection"
            selection.mkdir()
            contexts = (131072,)
            if mode != "baseline":
                for length in contexts:
                    stage = (size, "independent", length)
                    (selection / str(length)).symlink_to(post50.model(root, stage, step))
            for suite, task_names, fewshot in (
                    ("benchmark", None, 0), ("mmlu5", "mmlu", 5),
                    ("ppl", "prolong_ppl", 0)):
                directory = series / suite
                directory.mkdir()
                tp = 1 if size == "2b" else 4
                args = SimpleNamespace(
                    contexts=" ".join(map(str, contexts)), suite="all", tasks=task_names,
                    model_root=str(selection), model_template="{context}", tp=tp,
                    ruler_lengths=None, baseline_model=str(original) if mode == "baseline" else None)
                shards = build_shards(args, 12 // tp)
                settings = {"tp": tp, "memory_fraction": 0.8, "fewshot": fewshot, "limit": None}
                save(directory / "plan.json", {"nodes": ["pending-pbs-node"], "resume": True,
                                              "settings": settings,
                                              "shards": [asdict(s) for s in shards]})
                env = {"EVAL_DIR": str(Path(__file__).resolve().parent),
                       "EVAL_OUTPUT": str(directory), "EVAL_NODES": "1",
                       "EVAL_QUEUE": "debug", "MODEL_ROOT": str(selection),
                       "MODEL_TEMPLATE": "{context}", "CONTEXTS": args.contexts, "TP": str(tp),
                       "BASELINE_MODEL": args.baseline_model or "", "EVAL_FEWSHOT": str(fewshot),
                       "EVAL_TASKS": task_names or "", "EVAL_SUITE": "all",
                       "PPL_TASK_DIR": str(tasks / "tasks") if suite == "ppl" else "",
                       "POST50_EVAL_ROOT": "", "POST50_PPL_ROOT": "",
                       "COMPARISON_BASELINE": "", "COMPARISON_INDEPENDENT": ""}
                jobs.append({"size": size, "mode": mode, "step": step, "suite": suite,
                             "shards": len(shards), "environment": env})
    for before, after in zip(jobs, jobs[1:]):
        save(Path(before["environment"]["EVAL_OUTPUT"]) / "FOLLOWUP.json", after["environment"])
    save(output / "plan.json", {"jobs": jobs, "total_shards": sum(j["shards"] for j in jobs),
                                "submitted": False, "shared_16k": False})
    print(f"Prepared {len(jobs)} plans, no submission: {output}")
    return jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    prepare(parser.parse_args().root)
