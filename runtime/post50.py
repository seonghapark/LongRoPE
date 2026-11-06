"""Serial original-2B/20B to 128k sweep with permanent 50-step DCP and HF models."""

import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

from prolong_data import SAMPLING_POLICY

BASE = Path("/lus/flare/projects/datascience/seonghapark")
LAUNCH = Path(__file__).resolve().parent
HOME = LAUNCH.parent
PYTHON = BASE / "venv/bin/python"
STAGES = [(size, "independent", 131072) for size in ("2b", "20b")]
MILESTONES = tuple(range(50, 401, 50))


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def tag(stage):
    return "_".join(map(str, stage))


def model(root, stage, step=400):
    return root / "models" / tag(stage) / f"step-{step}"


def source(root, stage):
    size, mode, length = stage
    if mode == "cumulative":
        parent = (size, "shared" if length == 32768 else "cumulative", length // 2)
        path = model(root, parent)
        if not (path / "VERIFIED.json").is_file():
            raise FileNotFoundError(f"Unverified parent: {path}")
        return path
    name = ("agpt-2b-v2-256n-step-92859-safetensors" if size == "2b"
            else "agpt-20b-v2-512n-step-4000-safetensors")
    return BASE / name


def parameters(root, stage):
    size, _, length = stage
    cp = ({16384: 1, 32768: 1, 65536: 2, 131072: 4} if size == "2b"
          else {16384: 1, 32768: 2, 65536: 4, 131072: 4})[length]
    profile = root / "profiles" / f"{tag(stage)}.json"
    from longrope_profile import load

    load(profile, length=length, source=source(root, stage), searched=True)
    from prolong_data import stage_seed, validate_data

    data = root / "data/prolong"
    manifest = validate_data(data, model=source(root, stage), size=size)
    return {key: str(value) for key, value in {
        "AGPT_DATASET_DIR": data,
        "AGPT_DATA_SEED": stage_seed(root / "runs" / tag(stage)),
        "AGPT_TRAIN_TOKENS": sum(s["tokens"][size] for s in manifest["subsets"].values()),
        "LONGROPE_PROFILE": profile,
        "TORCHTITAN_ROOT": LAUNCH / "xpu_torchtitan/torchtitan_repo",
        "AGPT_SIZE": size, "AGPT_SEQ_LEN": length,
        "AGPT_TP": 2 if size == "2b" and length == 16384 else 4,
        "AGPT_CP": cp, "AGPT_COMPILE": int(size == "2b"),
        "AGPT_MODEL_IN": source(root, stage),
        "AGPT_FINAL_OUT": model(root, stage),
        "AGPT_RUN_DIR": root / "runs" / tag(stage),
        "AGPT_CKPT_INTERVAL": 10, "AGPT_KEEP_K": 3, "AGPT_KEEP_EVERY": 50,
        "AGPT_EXPORT": 0, "AGPT_AUTOCHAIN": 0, "AGPT_OOM_FALLBACK": 0,
        "AGPT_TRAIN_TIMEOUT": 3000, "AGPT_WALLTIME_S": 3600,
        "AGPT_SKIP_NONFINITE": 0, "AGPT_NAN_DIAG": 1,
        "AGPT_NAN_MAX_EVENTS": 4, "AGPT_LOSS_STD_ENABLED": 0,
        "AGPT_NAN_RETRY_ALWAYS": int(
            (root / "runs" / tag(stage) / "NAN_RETRY_UNTIL400").is_file()),
        **({"AGPT_CP_ROTATE_METHOD": "alltoall"} if size == "20b" else {}),
    }.items()}


def checkpoints(run):
    found = {}
    for metadata in run.glob("*/step-*/.metadata"):
        step = int(metadata.parent.name.removeprefix("step-"))
        if step in found:
            raise RuntimeError(f"Duplicate checkpoint step {step}")
        found[step] = metadata.parent
    return found


def next_action(root, index):
    stage = STAGES[index]
    run = root / "runs" / tag(stage)
    if (run / "HALT").exists():
        raise RuntimeError((run / "HALT").read_text())
    profile = root / "profiles" / f"{tag(stage)}.json"
    if not profile.exists():
        return {"index": index, "action": "search"}
    from longrope_profile import load

    load(profile, length=stage[2], source=source(root, stage), searched=True)
    available = checkpoints(run)
    for step in MILESTONES:
        if step in available and not (model(root, stage, step) / "VERIFIED.json").exists():
            return {"index": index, "action": "export", "step": step}
    if 400 in available:
        if not all((model(root, stage, step) / "VERIFIED.json").is_file()
                   for step in MILESTONES):
            raise RuntimeError("Missing milestone checkpoint/export")
        if index + 1 == len(STAGES):
            return None
        return next_action(root, index + 1)
    if (run / "DONE").exists():
        raise RuntimeError("Training ended before required step 400")
    return {"index": index, "action": "train"}


def submit(root, action):
    if action is None:
        (root / "DONE").write_text("2 original-model 128k stages; 16 verified HF exports and DCP milestones.\n")
        return
    index = action["index"]
    stage = STAGES[index]
    source_path = source(root, stage)
    if not (source_path / "config.json").is_file() or not list(source_path.glob("*.safetensors")):
        raise FileNotFoundError(source_path)
    result = subprocess.run(
        ["qsub", "-A", "AuroraGPT", "-q",
         "debug" if action["action"] == "search" else "debug-scaling",
         "-l", "select=1" if action["action"] == "search" else "select=4",
         "-l", "walltime=01:00:00", "-l", "filesystems=home:flare",
         "-N", f"lr_{stage[0]}_{stage[2]}", "-j", "oe",
         "-o", str(root / "logs"),
         "-v", f"LONGROPE_ROOT={root}", str(HOME / "job_longrope.sh")],
        capture_output=True, text=True, check=True)
    job = result.stdout.strip()
    if not job.split(".")[0].isdigit():
        raise RuntimeError(f"Unexpected qsub response {job!r}")
    state = dict(action, job=job, stage=stage)
    save(root / "state.json", state)
    with (root / "submissions.jsonl").open("a") as stream:
        stream.write(json.dumps(state) + "\n")
    print(json.dumps(state), flush=True)


def checked(command, log, *, cwd=None, timeout=900):
    with log.open("w") as stream:
        subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                       check=True, timeout=timeout)


def export(root, stage, step):
    run = root / "runs" / tag(stage)
    checkpoint = checkpoints(run)[step]
    destination = model(root, stage, step)
    temporary = destination.with_name(destination.name + ".pending")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    logs = root / "logs" / f"{tag(stage)}_step{step}"
    logs.mkdir(exist_ok=True)
    parent = source(root, stage)
    env = os.environ.copy()
    env.update(AGPT_OSEXIT_EXPERIMENT="0", AGPT_INDUCTOR_ISOLATE="0")
    env.update(parameters(root, stage))
    with (logs / "export.log").open("w") as stream:
        subprocess.run(
            [str(PYTHON), str(LAUNCH / "dcp_to_hf.py"), "--dcp-path", str(checkpoint),
             "--output", str(temporary), "--assets-from", str(parent), "--force", "--",
             "--module", "agpt", "--config", f"agpt_{stage[0]}_yarn",
             "--training.seq_len", str(stage[2])],
            cwd=LAUNCH / "xpu_torchtitan", env=env, stdout=stream,
            stderr=subprocess.STDOUT, check=True, timeout=2400)
    from longrope_profile import load, patch_model

    profile = load(root / "profiles" / f"{tag(stage)}.json", length=stage[2],
                   source=parent, searched=True)
    patch_model(temporary, profile)
    checked([str(PYTHON), "-I", str(LAUNCH / "verify_export_bf16.py"),
             str(parent), str(temporary)], logs / "tensors.log")
    report = (logs / "tensors.log").read_text()
    required = ("NaN/Inf tensors: none", "shape mismatches: [] (0)",
                "only in IN : []", "only in OUT: []", "sampled 24 tensors, 24 changed")
    if not all(text in report for text in required):
        raise RuntimeError(f"Tensor verification failed: {logs}")
    checked([str(PYTHON), str(HOME / "verify.py"), str(temporary)],
            logs / "rope.log")
    if "ALL MATCH" not in (logs / "rope.log").read_text():
        raise RuntimeError(f"RoPE verification failed: {logs}")
    save(temporary / "VERIFIED.json", {"stage": stage, "step": step,
                                     "checkpoint": str(checkpoint), "parent": str(parent)})
    (temporary / "TRAINED_STEPS").write_text(f"step-{step} of 400, from {checkpoint}\n")
    temporary.rename(destination)


def main():
    root = Path(os.environ["LONGROPE_ROOT"]).resolve()
    root.relative_to(HOME / "experiments")
    root.mkdir(exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    with (root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "HALT").exists():
            raise RuntimeError((root / "HALT").read_text())
        if sys.argv[1] == "start":
            if (root / "state.json").exists():
                raise FileExistsError(root / "state.json")
            plan = json.loads((root / "plan.json").read_text())
            if plan.get("stages") != [list(s) for s in STAGES]:
                raise ValueError("This controller requires the new two-stage ProLong-128k plan")
            from prolong_data import validate_data

            for stage in STAGES:
                validate_data(root / "data/prolong", model=source(root, stage),
                              size=stage[0], verify_hashes=True)
            save(root / "execution_plan.json", {"stages": STAGES, "milestones": MILESTONES,
                                               "shared_16k": False, "steps": 400,
                                               "sampling": SAMPLING_POLICY})
            submit(root, next_action(root, 0))
            return
        if sys.argv[1] != "run":
            raise ValueError("Expected start or run")
        state = json.loads((root / "state.json").read_text())
        if state["job"] != os.environ["PBS_JOBID"]:
            raise RuntimeError("Stale/duplicate job refused")
        stage = STAGES[state["index"]]
        try:
            if state["action"] == "search":
                result = subprocess.run(
                    [str(BASE / "llm_evaluation/.venv-post2b/bin/python"),
                     str(HOME / "search.py"), "--root", str(root),
                     "--stage", str(state["index"]), "--budget-seconds", "2700"])
                if result.returncode not in (0, 124):
                    raise RuntimeError(f"Search failed rc={result.returncode}")
            elif state["action"] == "train":
                env = {k: v for k, v in os.environ.items() if not k.startswith("AGPT_")}
                env.update(parameters(root, stage))
                run = root / "runs" / tag(stage)
                history = run / "NAN_HISTORY"
                before = history.read_text().splitlines() if history.exists() else []
                result = subprocess.run(["bash", str(LAUNCH / "job_agpt_post.sh")], env=env)
                after = history.read_text().splitlines() if history.exists() else []
                if result.returncode not in (0, 124) and len(after) <= len(before):
                    raise RuntimeError(f"Training failed rc={result.returncode}")
            else:
                export(root, stage, state["step"])
            submit(root, next_action(root, state["index"]))
        except Exception as error:
            (root / "HALT").write_text(f"{type(error).__name__}: {error}\n")
            raise


if __name__ == "__main__":
    main()
