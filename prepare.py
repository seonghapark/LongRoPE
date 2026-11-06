"""Snapshot the working launcher and prepare an isolated, unsubmitted sweep."""

import hashlib
import argparse
import json
import shutil
from pathlib import Path

HOME = Path(__file__).resolve().parent
LAUNCHER = HOME.parent / "xpu_launcher"
ROOT = HOME / "experiments" / "prolong128k"
STAGES = [(size, "independent", 131072) for size in ("2b", "20b")]


def snapshot(resume=False):
    runtime = HOME / "runtime"
    if runtime.exists() and (not resume or (HOME / "SOURCE_SNAPSHOT.json").exists()):
        raise FileExistsError(f"Refusing to overwrite the runtime: {runtime}")
    runtime.mkdir(exist_ok=resume)
    files = ("post50.py", "job_agpt_post.sh", "run_train.sh", "dcp_to_hf.py",
             "verify_export_bf16.py", "agg_resources.py")
    for name in files:
        shutil.copy2(LAUNCHER / name, runtime / name)
    patterns = shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "outputs", "core.*")

    def ignore(path, names):
        return set(patterns(path, names)) | {
            name for name in names if (Path(path) / name).is_symlink()}
    for name in ("src", "inductor_iso", "osexit_exp"):
        shutil.copytree(LAUNCHER / name, runtime / name, ignore=ignore,
                        symlinks=True, dirs_exist_ok=resume)
    titan = runtime / "xpu_torchtitan"
    titan.mkdir(exist_ok=resume)
    for name in ("titan_train.py", "run_train_agpt.sh", "run_train_torchtitan.sh",
                 "calculate_agpt_params.sh"):
        shutil.copy2(LAUNCHER / "xpu_torchtitan" / name, titan / name)
    source = LAUNCHER / "xpu_torchtitan/torchtitan_repo"
    repo = titan / "torchtitan_repo"
    repo.mkdir(exist_ok=resume)
    for name in ("torchtitan", "tests"):
        shutil.copytree(source / name, repo / name, ignore=ignore,
                        symlinks=True, dirs_exist_ok=resume)
    for name in ("LICENSE", "pyproject.toml", "requirements.txt"):
        shutil.copy2(source / name, repo / name)
    hashes = {str(p.relative_to(runtime)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in runtime.rglob("*") if p.is_file()}
    (HOME / "SOURCE_SNAPSHOT.json").write_text(json.dumps({
        "source": str(LAUNCHER), "note": "working files, including existing local fixes",
        "sha256_before_longrope_changes": hashes}, indent=2) + "\n")


def prepare():
    from runtime.prolong_data import SAMPLING_POLICY, SPLIT_POLICY

    ROOT.mkdir(parents=True, exist_ok=False)
    for name in ("profiles", "search", "logs", "runs", "models", "evaluation"):
        (ROOT / name).mkdir()
    jobs = []
    for stage in STAGES:
        name = "_".join(map(str, stage))
        jobs.append({"stage": stage, "profile": str(ROOT / "profiles" / f"{name}.json"),
                     "status": "NOT_SEARCHED_NOT_SUBMITTED"})
    (ROOT / "plan.json").write_text(json.dumps({
        "rope_type": "longrope", "stages": STAGES, "jobs": jobs, "steps": 400,
        "milestones": list(range(50, 401, 50)), "permanent_hf_exports": 16,
        "dataset": "princeton-nlp/prolong-data-512K",
        "sampling": SAMPLING_POLICY, "heldout_partition": SPLIT_POLICY,
        "data_status": "NOT_CONVERTED", "retained_token_candidates": [0, 1, 4, 16, 64, 256],
        "search_split": "PG19 validation", "evaluation_split": "fixed ProLong held-out",
        "search_defaults": {"samples": 4, "population": 8, "generations": 8},
        "training_queue": "debug-scaling", "training_nodes": 4,
        "walltime": "01:00:00", "submitted": False,
    }, indent=2) + "\n")
    print(f"Prepared 2 original-model 128k stages / 16 milestones, no jobs submitted: {ROOT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-128k", action="store_true",
                        help="Use the existing runtime; never overwrite its snapshot")
    parser.add_argument("--resume-snapshot", action="store_true")
    args = parser.parse_args()
    if not args.prepare_128k:
        snapshot(args.resume_snapshot)
    prepare()
