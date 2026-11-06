"""Prepare an isolated 14-stage YaRN/ProLong sweep after verified LongRoPE completion."""

import argparse
import fcntl
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

HOME = Path(__file__).resolve().parent
BASE = HOME.parent
RUNTIME = HOME / "runtime"
sys.path.insert(0, str(RUNTIME))
from prolong_data import SAMPLING_POLICY, stage_seed, validate_data  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "yarn_prolong_core", BASE / "xpu_launcher/post50.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
ROOT = HOME / "experiments/yarn_prolong_post50"
LONGROPE_ROOT = HOME / "experiments/prolong128k"
original_parameters = core.parameters


def prerequisite(root):
    plan = json.loads((root / "plan.json").read_text())
    if (plan["stages"] != [list(stage) for stage in core.STAGES]
            or plan["sampling"] != SAMPLING_POLICY):
        raise ValueError("Incompatible YaRN ProLong plan or sampling policy")
    parent = Path(plan["after_longrope"]).resolve()
    parent.relative_to(HOME / "experiments")
    if not (parent / "DONE").is_file():
        raise RuntimeError("LongRoPE training/export sweep has not completed")
    for size in ("2b", "20b"):
        for step in core.MILESTONES:
            model = parent / "models" / f"{size}_independent_131072" / f"step-{step}"
            if not (model / "VERIFIED.json").is_file():
                raise FileNotFoundError(f"Missing verified LongRoPE export: {model}")
    return plan


def parameters(root, stage):
    values = original_parameters(root, stage)
    manifest = validate_data(root / "data/prolong", core.source(root, stage), stage[0])
    values.update(
        AGPT_DATASET_NAME="prolong", AGPT_DATASET_DIR=str(root / "data/prolong"),
        AGPT_DATA_SEED=str(stage_seed(root / "runs" / core.tag(stage))),
        AGPT_TRAIN_TOKENS=str(sum(s["tokens"][stage[0]]
                                  for s in manifest["subsets"].values())))
    return values


def submit(root, action):
    prerequisite(root)
    if not (root / "GPU_APPROVED.json").is_file():
        raise PermissionError("Explicit start --approve-gpu is required")
    if action is None:
        (root / "DONE").write_text("14 YaRN ProLong stages; 112 verified HF exports.\n")
        return
    stage = core.STAGES[action["index"]]
    source = core.source(root, stage)
    if not (source / "config.json").is_file() or not list(source.glob("*.safetensors")):
        raise FileNotFoundError(f"Missing YaRN source model: {source}")
    validate_data(root / "data/prolong", source, stage[0],
                  verify_hashes=True)
    result = subprocess.run(
        ["qsub", "-A", "AuroraGPT", "-q", "debug-scaling", "-l", "select=4",
         "-l", "walltime=01:00:00", "-l", "filesystems=home:flare",
         "-N", f"yp_{stage[0]}_{stage[2]}", "-j", "oe", "-o", str(root / "logs"),
         "-v", f"POST50_ROOT={root}", str(HOME / "job_yarn_prolong.sh")],
        capture_output=True, text=True, check=True)
    job = result.stdout.strip()
    if not job.split(".")[0].isdigit():
        raise RuntimeError(f"Unexpected qsub response: {job!r}")
    state = dict(action, stage=stage, job=job)
    core.save(root / "state.json", state)
    with (root / "submissions.jsonl").open("a") as stream:
        stream.write(json.dumps(state) + "\n")
    print(json.dumps(state), flush=True)


core.parameters = parameters
core.submit = submit


def prepare(root, after=LONGROPE_ROOT):
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    (root / "data").mkdir()
    (root / "data/prolong").symlink_to(after.resolve() / "data/prolong")
    core.save(root / "plan.json", {
        "stages": core.STAGES, "milestones": core.MILESTONES, "steps": 400,
        "shared_16k": True, "exports": 112, "after_longrope": str(after.resolve()),
        "sampling": SAMPLING_POLICY, "dataset": "princeton-nlp/prolong-data-512K",
        "submitted": False, "gpu_approval_required": True,
        "initial_models": "original 2B/20B; not LongRoPE weights",
    })
    print(f"Prepared 14 stages / 112 exports; no submission: {root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "start", "run"))
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("POST50_ROOT", ROOT)))
    parser.add_argument("--approve-gpu", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    root.relative_to(HOME / "experiments")
    if args.action == "prepare":
        prepare(root)
        return
    os.environ.pop("LONGROPE_PROFILE", None)
    os.environ["TORCHTITAN_ROOT"] = str(BASE / "xpu_launcher/xpu_torchtitan/torchtitan_repo")
    os.environ["PYTHONPATH"] = str(RUNTIME)
    os.environ["POST50_ROOT"] = str(root)
    if args.action == "run":
        sys.argv = [sys.argv[0], "run"]
        core.main()
        return
    if not args.approve_gpu:
        raise PermissionError("GPU submission requires explicit start --approve-gpu")
    with (root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prerequisite(root)
        if (root / "state.json").exists() or (root / "HALT").exists():
            raise FileExistsError("Existing YaRN state/HALT; inspect before retrying")
        for size in ("2b", "20b"):
            validate_data(root / "data/prolong", core.source(root, (size, "shared", 16384)),
                          size, verify_hashes=True)
        core.save(root / "GPU_APPROVED.json", {"command": "start --approve-gpu"})
        submit(root, core.next_action(root, 0))


if __name__ == "__main__":
    main()
