"""Plan and run independent lm-eval/vLLM shards on Aurora nodes."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import re
import runpy
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


CONTEXTS = (16384, 32768, 65536, 131072)
STANDARD = (
    "hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande",
    "lambada_openai", "boolq", "wikitext",
)
LONGBENCH = (
    "longbench_multifieldqa_en", "longbench_musique",
    "longbench_gov_report", "longbench_lcc",
)
NIAH = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multiquery", "niah_multivalue",
)
RULER = NIAH + ("ruler_vt", "ruler_cwe", "ruler_fwe", "ruler_qa_squad", "ruler_qa_hotpot")
PERPLEXITY = ("pg19_ppl", "wikitext_ppl", "prolong_ppl")
REMOTE_ENV = (
    "EVAL_DIR", "EVAL_VENV", "EVAL_FRAMEWORKS", "HARNESS_DIR", "PYTHONPATH",
    "HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE",
    "HF_HUB_ENABLE_HF_TRANSFER", "NLTK_DATA", "LM_EVAL_URL_CACHE",
    "ZE_FLAT_DEVICE_HIERARCHY", "CCL_PROCESS_LAUNCHER",
    "VLLM_WORKER_MULTIPROC_METHOD", "TOKENIZERS_PARALLELISM",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "no_proxy", "NO_PROXY",
    "PPL_TASK_DIR",
)


@dataclass
class Shard:
    context: int
    model: str
    task: str
    length: int | None
    slot: int = 0

    @property
    def tag(self) -> str:
        suffix = f"_len{self.length}" if self.length else ""
        return f"{self.context}/{self.task}{suffix}"


def words(value: str) -> list[str]:
    return value.replace(",", " ").split()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def build_shards(args: argparse.Namespace, slots: int) -> list[Shard]:
    tasks = words(args.tasks) if args.tasks else {
        "standard": list(STANDARD),
        "long": ["ruler", *LONGBENCH],
        "all": [*STANDARD, "ruler", *LONGBENCH],
    }[args.suite]
    if not tasks or any(not re.fullmatch(r"[\w-]+", task, re.ASCII) for task in tasks):
        raise ValueError("Tasks must be exact task names, not patterns or shell expressions.")
    shards: dict[str, Shard] = {}
    for context in dict.fromkeys(map(int, words(args.contexts))):
        if context not in CONTEXTS:
            raise ValueError(f"Unsupported post-trained context: {context}")
        baseline = getattr(args, "baseline_model", None)
        template = getattr(args, "model_template", "agpt-2b-v2-256n-post-{context}")
        model = (Path(baseline) if baseline else
                 Path(args.model_root) / template.format(context=context)).resolve()
        config = json.loads((model / "config.json").read_text())
        if baseline and config["max_position_embeddings"] < context:
            raise ValueError(f"{model}: evaluation context {context} exceeds its configured window")
        if not baseline and config["max_position_embeddings"] != context:
            raise ValueError(f"{model}: expected max_position_embeddings={context}")
        if not baseline:
            from longrope_profile import load, scaling

            profile = load(model / "LONGROPE_PROFILE.json", length=context, searched=True)
            if config.get("rope_scaling") != scaling(profile):
                raise ValueError(f"{model}: config differs from searched LongRoPE profile")
            if (config.get("longrope_retained_tokens") != profile["retained_tokens"]
                    or config.get("longrope_requires_custom_runtime") is not True):
                raise ValueError(f"{model}: missing/mismatched retained-prefix runtime metadata")
            if not (model / "VERIFIED.json").is_file():
                raise FileNotFoundError(f"Unverified LongRoPE model: {model}")
        if not (model / "tokenizer.json").is_file() or not list(model.glob("*.safetensors")):
            raise ValueError(f"{model}: tokenizer or safetensors weights missing")
        for key in ("num_attention_heads", "num_key_value_heads"):
            if config[key] % args.tp:
                raise ValueError(f"TP={args.tp} must divide {key}={config[key]}")
        lengths = list(map(int, words(args.ruler_lengths))) if args.ruler_lengths else [
            4096 * 2**i for i in range(6) if 4096 * 2**i <= context
        ]
        if any(length < 4096 or length > context for length in lengths):
            raise ValueError(f"RULER lengths must be between 4096 and {context}")
        for task in tasks:
            expanded = RULER if task == "ruler" else NIAH if task == "niah" else (task,)
            for subtask in expanded:
                for length in lengths if subtask in RULER + PERPLEXITY else [None]:
                    shard = Shard(context, str(model), subtask, length)
                    shards[shard.tag] = shard
    if not shards:
        raise ValueError("No evaluation shards selected.")
    loads = [0] * slots
    ordered = sorted(shards.values(), key=lambda s: -(s.length or (
        s.context if s.task.startswith("longbench_") else 4096
    )))
    for shard in ordered:
        shard.slot = min(range(slots), key=loads.__getitem__)
        loads[shard.slot] += shard.length or (
            shard.context if shard.task.startswith("longbench_") else 4096
        )
    return ordered


def command(shard: Shard, settings: dict, output: Path) -> list[str]:
    max_len = shard.length or (
        shard.context if shard.task.startswith("longbench_") else min(4096, shard.context)
    )
    model_args = (
        f"pretrained={shard.model},tokenizer={shard.model},dtype=bfloat16,"
        f"tensor_parallel_size={settings['tp']},data_parallel_size=1,"
        f"gpu_memory_utilization={settings['memory_fraction']},"
        f"max_model_len={max_len},max_num_batched_tokens={max_len},"
        "max_num_seqs=1,enforce_eager=True,enable_prefix_caching=False,"
        "worker_cls=longrope_vllm.RetainedXPUWorker"
    )
    if shard.task.startswith("longbench_"):
        model_args += ",truncation_side=middle"
    if shard.task in PERPLEXITY:
        model_args = model_args.replace(
            f"max_num_batched_tokens={max_len},",
            f"max_num_batched_tokens={min(max_len, 4096)},")
        model_args += ",add_bos_token=False,enable_chunked_prefill=True"
    cmd = [
        "--model", "vllm", "--model_args", model_args,
        "--tasks", shard.task, "--num_fewshot", str(settings["fewshot"]),
        "--batch_size", "1", "--seed", "0,1234,1234,1234",
        "--output_path", str(output), "--log_samples",
        "--use_cache", str(output / "responses"),
    ]
    if shard.task in PERPLEXITY:
        cmd[cmd.index("--model") + 1] = "ppl_vllm"
        cmd += ["--include_path", os.environ["PPL_TASK_DIR"]]
    elif shard.length:
        cmd += ["--metadata", json.dumps({"max_seq_lengths": [shard.length]})]
    if settings["limit"] is not None:
        cmd += ["--limit", str(settings["limit"])]
    return cmd


def preflight(shards: list[Shard]) -> None:
    import lm_eval
    from lm_eval.tasks import TaskManager

    expected = Path(os.environ["HARNESS_DIR"]) / "lm_eval"
    if Path(lm_eval.__file__).resolve().parent != expected.resolve():
        raise RuntimeError(f"Wrong lm_eval source: {lm_eval.__file__}; expected {expected}")
    modules = ["vllm", "lm_eval.models.vllm_causallms"]
    if any(s.task in RULER for s in shards):
        modules += ["lm_eval.tasks.ruler.prepare_niah"]
    if any(s.task.startswith("longbench_") for s in shards):
        modules += ["lm_eval.tasks.longbench.metrics"]
    if any(s.task in PERPLEXITY for s in shards):
        modules += ["ppl_vllm"]
    for module in modules:
        importlib.import_module(module)
    task_manager = TaskManager(include_path=os.environ.get("PPL_TASK_DIR") or None)
    missing = {s.task for s in shards} - set(task_manager.all_tasks)
    if missing:
        raise ValueError(f"Unregistered harness tasks: {sorted(missing)}")
    if any(s.task == "ruler_qa_hotpot" for s in shards):
        cache = Path(os.environ["LM_EVAL_URL_CACHE"]) / "hotpot_dev_distractor_v1.json"
        if not cache.is_file():
            raise FileNotFoundError(f"RULER HotpotQA needs the local cache: {cache}")
    print(f"[INFO] Harness: {lm_eval.__file__}", flush=True)


def run_shard(manifest: Path, index: int) -> None:
    plan = json.loads(manifest.read_text())
    shard = Shard(**plan["shards"][index])
    if shard.task in PERPLEXITY:
        importlib.import_module("ppl_vllm")
    output = manifest.parent / shard.tag
    sys.argv = ["lm_eval", *command(shard, plan["settings"], output)]
    runpy.run_module("lm_eval", run_name="__main__")


def worker(manifest: Path, slot: int) -> int:
    plan = json.loads(manifest.read_text())
    settings = plan["settings"]
    local_slot = slot % (12 // settings["tp"])
    first = local_slot * settings["tp"]
    os.environ["ZE_AFFINITY_MASK"] = ",".join(map(str, range(first, first + settings["tp"])))
    os.environ["VLLM_PORT"] = str(29500 + local_slot * 200)
    # Avoid compiler/cache file contention between independent engines.
    cache = manifest.parent / "cache" / f"slot_{slot}"
    os.environ["VLLM_CACHE_ROOT"] = str(cache / "vllm")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
    status_path = manifest.parent / f"status_{slot}.json"
    statuses = json.loads(status_path.read_text()) if plan["resume"] and status_path.is_file() else {}
    for index, item in enumerate(plan["shards"]):
        shard = Shard(**item)
        if shard.slot != slot:
            continue
        output = manifest.parent / shard.tag
        if statuses.get(shard.tag) == 0 and list(output.rglob("results_*.json")):
            print(f"[SKIP] Already completed: {shard.tag}", flush=True)
            continue
        output.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--run-shard", str(manifest), str(index)]
        print(f"[START] slot={slot} tiles={os.environ['ZE_AFFINITY_MASK']} {shard.tag}", flush=True)
        with (output / "run.log").open("w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
        if rc == 0 and not list(output.rglob("results_*.json")):
            print(f"[ERROR] {shard.tag}: process exited 0 without results", flush=True)
            rc = 1
        statuses[shard.tag] = rc
        write_json(status_path, statuses)
        print(f"[END] {shard.tag} rc={rc}", flush=True)
    return int(any(statuses.values()))


def summarize(manifest: Path, report_missing: bool = True) -> bool:
    plan = json.loads(manifest.read_text())
    statuses = {}
    for path in manifest.parent.glob("status_*.json"):
        statuses.update(json.loads(path.read_text()))
    rows = []
    complete = True
    for item in plan["shards"]:
        shard = Shard(**item)
        results = list((manifest.parent / shard.tag).rglob("results_*.json"))
        if statuses.get(shard.tag) != 0 or not results:
            if report_missing:
                print(f"[ERROR] Missing/failed shard: {shard.tag}", file=sys.stderr)
            complete = False
            continue
        path = max(results, key=lambda p: p.stat().st_mtime_ns)
        result = json.loads(path.read_text())
        metrics_by_task = dict(result["results"])
        if shard.task in PERPLEXITY:
            value = metrics_by_task.get(shard.task, {}).get("token_perplexity,none")
            counts = result.get("n-samples", {}).get(shard.task, {})
            if (type(value) not in (int, float) or not math.isfinite(value) or value < 1
                    or counts.get("original", 0) <= 0
                    or counts.get("effective") != counts["original"]
                    or result.get("n-shot", {}).get(shard.task) != 0):
                raise ValueError(f"Invalid/incomplete full-test perplexity results: {path}")
        if shard.task == "mmlu":
            subjects = [name for name in result.get("configs", {}) if name.startswith("mmlu_")]
            if len(subjects) != 57 or "acc,none" not in result.get("groups", {}).get("mmlu", {}):
                raise ValueError(f"Incomplete MMLU results (expected 57 subjects and aggregate): {path}")
            if any("acc,none" not in metrics_by_task.get(name, {})
                   for name in subjects):
                raise ValueError(f"Missing MMLU subject accuracy: {path}")
            if any(result.get("n-shot", {}).get(name) != plan["settings"]["fewshot"]
                   for name in subjects):
                raise ValueError(f"MMLU few-shot setting mismatch: {path}")
            metrics_by_task["mmlu"] = result["groups"]["mmlu"]
        for task, metrics in metrics_by_task.items():
            for metric, value in metrics.items():
                if (type(value) in (int, float) and "_stderr" not in metric
                        and not (metric.split(",")[0].isdigit() and value == -1)):
                    rows.append((shard.context, task, shard.length or "", metric, value))
    temporary = manifest.parent / "summary.csv.tmp"
    with temporary.open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(("context", "task", "ruler_length", "metric", "value"))
        writer.writerows(rows)
    temporary.replace(manifest.parent / "summary.csv")
    return complete


def plot_summary(manifest: Path) -> None:
    from report import plot_plan

    plot_plan(manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=2, metavar=("PLAN", "SLOT"))
    parser.add_argument("--run-shard", nargs=2, metavar=("PLAN", "INDEX"))
    parser.add_argument("--report", type=Path, metavar="PLAN",
                        help="Collect completed results and plot a running/finished plan.")
    parser.add_argument("--model-root", default=os.environ.get(
        "MODEL_ROOT", str(Path(__file__).resolve().parent.parent)))
    parser.add_argument("--model-template", default=os.environ.get(
        "MODEL_TEMPLATE", "agpt-2b-v2-256n-post-{context}"))
    parser.add_argument("--baseline-model", default=os.environ.get("BASELINE_MODEL"),
                        help="Evaluate unchanged original HF weights/config at the selected windows.")
    parser.add_argument("--contexts", default=os.environ.get("CONTEXTS", "16384 32768 65536 131072"))
    parser.add_argument("--suite", choices=("standard", "long", "all"),
                        default=os.environ.get("EVAL_SUITE", "all"))
    parser.add_argument("--tasks", default=os.environ.get("EVAL_TASKS"))
    parser.add_argument("--ruler-lengths", default=os.environ.get("RULER_LENGTHS"))
    parser.add_argument("--tp", type=int, default=int(os.environ.get("TP", "1")))
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    parser.add_argument("--fewshot", type=int, default=int(os.environ.get("EVAL_FEWSHOT", "0")))
    parser.add_argument("--limit", type=float, default=os.environ.get("LIMIT"))
    parser.add_argument("--nodefile", default=os.environ.get("PBS_NODEFILE"))
    parser.add_argument("--nodes", type=int, choices=(1, 2),
                        default=int(os.environ.get("EVAL_NODES", "2")))
    parser.add_argument("--output", type=Path, default=os.environ.get("EVAL_OUTPUT"))
    parser.add_argument("--dry-run", action="store_true", default=os.environ.get("DRY_RUN") == "1")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--resume", action="store_true", default=os.environ.get("RESUME") == "1")
    args = parser.parse_args()
    if args.report:
        complete = summarize(args.report, report_missing=False)
        plot_summary(args.report)
        print(f"[INFO] Report refreshed; all shards complete={complete}")
        return 0
    if args.worker:
        return worker(Path(args.worker[0]), int(args.worker[1]))
    if args.run_shard:
        run_shard(Path(args.run_shard[0]), int(args.run_shard[1]))
        return 0
    if args.tp not in (1, 2, 4) or not 0 < args.memory_fraction < 1 or args.fewshot < 0:
        parser.error("TP must be 1, 2 or 4; memory fraction in (0,1); fewshot >= 0")
    if args.limit is not None and not (
            0 < args.limit < 1 or args.limit >= 1 and args.limit.is_integer()):
        parser.error("limit must be a positive integer count or a fraction in (0,1)")
    if args.resume and args.output is None:
        parser.error("--resume requires --output / EVAL_OUTPUT pointing at the original run")
    if args.nodefile:
        nodes = list(dict.fromkeys(Path(args.nodefile).read_text().split()))
        if len(nodes) != args.nodes:
            parser.error(f"Expected {args.nodes} distinct allocated nodes; found {len(nodes)}")
    elif args.dry_run or args.check:
        nodes = [f"dry-node-{n}" for n in range(args.nodes)]
    else:
        parser.error("PBS_NODEFILE is required; submit run_posttrained_2b_eval.sh with qsub")
    shards = build_shards(args, len(nodes) * (12 // args.tp))
    if not args.dry_run:
        preflight(shards)
    if args.check:
        print(f"[OK] Models, task registrations and imports checked ({len(shards)} shards).")
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = args.output or Path(__file__).resolve().parent / "outputs" / f"post2b_{stamp}"
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    manifest = root / "plan.json"
    plan = {
        "nodes": nodes, "resume": args.resume,
        "settings": {"tp": args.tp, "memory_fraction": args.memory_fraction,
                                   "fewshot": args.fewshot, "limit": args.limit},
        "shards": [asdict(s) for s in shards],
    }
    if args.resume:
        previous = json.loads(manifest.read_text())
        if any(previous[key] != plan[key] for key in ("settings", "shards")):
            raise ValueError("Resume requires the same models, tasks, TP, limit and fewshot settings.")
    write_json(manifest, plan)
    slots = sorted({s.slot for s in shards})
    print(f"[INFO] {len(shards)} shards, {len(slots)} active slots, TP={args.tp}; plan={manifest}")
    if args.dry_run:
        print("[INFO] Dry run only: no SSH, GPU work, or dataset downloads.")
        return 0
    (root / "DONE").unlink(missing_ok=True)
    summarize(manifest, report_missing=False)
    exports = "\n".join(f"export {key}={shlex.quote(os.environ[key])}"
                        for key in REMOTE_ENV if key in os.environ)
    driver = root / "worker.sh"
    driver.write_text(
        "#!/bin/bash\nset -eo pipefail\n"
        + f"module load {shlex.quote(os.environ['EVAL_FRAMEWORKS'])}\n"
        + f"source {shlex.quote(os.environ['EVAL_VENV'] + '/bin/activate')}\n"
        + exports + "\n"
        + f"\nexec python -u {shlex.quote(str(Path(__file__).resolve()))}"
        + f' --worker {shlex.quote(str(manifest))} "$1"\n'
    )
    failed = False
    processes = []
    try:
        for slot in slots:
            node = nodes[slot // (12 // args.tp)]
            with (root / f"slot_{slot}.log").open("w") as log:
                process = subprocess.Popen(
                    ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", node,
                     shlex.join(["bash", "-l", str(driver), str(slot)])],
                    stdout=log, stderr=subprocess.STDOUT,
                )
            processes.append(process)
        pending = list(processes)
        while pending:
            for process in pending.copy():
                try:
                    rc = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    continue
                failed = failed or rc != 0
                pending.remove(process)
            summarize(manifest, report_missing=False)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
    complete = summarize(manifest)
    if failed or not complete:
        plot_summary(manifest)
        print(f"[ERROR] Partial evaluation; inspect slot logs and summary.csv in {root}", file=sys.stderr)
        return 1
    (root / "DONE").write_text(datetime.now(timezone.utc).isoformat() + "\n")
    plot_summary(manifest)
    print(f"[DONE] {root / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
