"""Reserve an afterany continuation before the one-hour PBS allocation expires."""

import json
import os
import subprocess
import sys
import time
import fcntl
from pathlib import Path

from eval_posttrained_2b import plot_summary, summarize, write_json


def progress(root: Path) -> tuple[int, int]:
    successes = sum(value == 0 for path in root.glob("status_*.json")
                    for value in json.loads(path.read_text()).values())
    # SQLite caches grow as partial inference responses are committed.
    cached_bytes = sum(path.stat().st_size for path in root.rglob("responses*")
                       if path.is_file())
    return successes, cached_bytes


def submit(root: Path, job: str, overrides: dict | None = None) -> str:
    environment = dict(os.environ)
    environment.update(overrides or {})
    names = [
        "EVAL_DIR", "EVAL_VENV", "EVAL_FRAMEWORKS", "HARNESS_DIR", "EVAL_OUTPUT",
        "CONTEXTS", "EVAL_SUITE", "EVAL_TASKS", "RULER_LENGTHS", "TP", "LIMIT",
        "HF_HOME", "HF_DATASETS_CACHE", "NLTK_DATA", "LM_EVAL_URL_CACHE",
        "BASELINE_MODEL",
        "EVAL_NODES", "MODEL_ROOT", "MODEL_TEMPLATE",
        "COMPARISON_BASELINE", "COMPARISON_INDEPENDENT",
        "POST50_EVAL_ROOT", "EVAL_QUEUE", "EVAL_FEWSHOT",
        "POST50_PPL_ROOT", "PPL_TASK_DIR",
    ]
    exports = [f"{name}={environment[name]}" for name in names if name in environment]
    if any("," in value for value in exports):
        raise ValueError("Auto-chain requires space-separated lists, not commas, in environment values.")
    exports += ["RESUME=1", "AUTO_CHAIN=1"]
    script = Path(__file__).with_name("run_posttrained_2b_eval.sh")
    result = subprocess.run(
        ["qsub", "-N", "eval_base2b" if environment.get("BASELINE_MODEL") else "eval_post2b",
         "-q", environment.get("EVAL_QUEUE", "debug"),
         "-l", f"select={environment.get('EVAL_NODES', '2')}",
         "-W", f"depend=afterany:{job}",
         "-v", ",".join(exports), "-j", "oe", "-o", str(root / f"chain_after_{job}.log"),
         "-m", "e", str(script)],
        check=True, capture_output=True, text=True,
    )
    next_job = result.stdout.strip()
    if not next_job or not next_job.split(".")[0].isdigit():
        raise RuntimeError(f"Unexpected qsub response: {result.stdout!r} {result.stderr!r}")
    with (root / "chain.log").open("a") as stream:
        stream.write(f"{job} -> {next_job}\n")
    print(f"[CHAIN] Reserved {next_job} after {job}", flush=True)
    return next_job


def submit_followup(root: Path, job: str) -> None:
    with (root / "followup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _submit_followup(root, job)


def _submit_followup(root: Path, job: str) -> None:
    request = root / "FOLLOWUP.json"
    receipt = root / "FOLLOWUP_SUBMITTED.json"
    if not request.exists() or receipt.exists():
        return
    overrides = json.loads(request.read_text())
    destination = Path(overrides["EVAL_OUTPUT"])
    if not (destination / "plan.json").is_file():
        raise FileNotFoundError(f"Follow-up evaluation plan missing: {destination}")
    next_job = submit(destination, job, overrides)
    write_json(receipt, {"job": next_job, "output": str(destination)})
    print(f"[FOLLOWUP] Baseline evaluation reserved: {next_job}", flush=True)


def wait_followup(root: Path, timeout: float = 86400) -> int:
    deadline = time.monotonic() + timeout
    print(f"[WAIT] Preserving current evaluation; baseline will be submitted after DONE: {root}",
          flush=True)
    while time.monotonic() < deadline:
        if (root / "FOLLOWUP_SUBMITTED.json").exists():
            print((root / "FOLLOWUP_SUBMITTED.json").read_text(), flush=True)
            return 0
        if (root / "HALT").exists():
            raise RuntimeError(f"Post-trained evaluation halted; baseline not submitted: {root}")
        if (root / "DONE").exists():
            state = json.loads((root / "chain_state.json").read_text())
            try:
                submit_followup(root, state["job"])
            except subprocess.CalledProcessError as error:
                print(f"[ERROR] Follow-up qsub failed; retrying in 30s: {error.stderr}", flush=True)
            else:
                return 0
        time.sleep(30)
    raise TimeoutError("Baseline submission watcher expired after 24 hours.")


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--wait-followup":
        return wait_followup(Path(sys.argv[2]))
    root = Path(os.environ["EVAL_OUTPUT"]).resolve()
    job = os.environ["PBS_JOBID"]
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "chain_state.json"
    before = progress(root)
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    stalled = previous.get("stalled", 0) + 1 if previous.get("progress") == list(before) else 0
    if stalled >= 3:
        (root / "HALT").write_text("Three consecutive allocations without completed-shard/cache growth.\n")
        print("[ERROR] No progress for three allocations; automatic continuation halted.", file=sys.stderr)
        return 1
    write_json(state_path, {"progress": before, "stalled": stalled, "job": job})
    manifest = root / "plan.json"
    evaluator = Path(__file__).with_name("eval_posttrained_2b.py")
    process = subprocess.Popen([sys.executable, "-u", str(evaluator), *sys.argv[1:]])
    successor = None
    try:
        try:
            rc = process.wait(timeout=3000)
        except subprocess.TimeoutExpired:
            # Reserve while there is still time; afterany prevents concurrent writes.
            successor = submit(root, job)
            if manifest.exists():
                plot_summary(manifest)
            rc = process.wait()
        complete = manifest.exists() and summarize(manifest, report_missing=False)
        if complete and (root / "DONE").exists() and rc == 0:
            if successor:
                subprocess.run(["qdel", successor], check=True)
            submit_followup(root, job)
            print(f"[ALL SUCCESS] Verified every shard; CSV and plots: {root}", flush=True)
            return 0
        if not successor:
            submit(root, job)
        print(f"[CHAIN] Evaluation incomplete (rc={rc}); successor will resume.", flush=True)
        return 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()


if __name__ == "__main__":
    sys.exit(main())
