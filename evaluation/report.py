"""Window-preserving LongRoPE plots and cross-checkpoint CSV collection."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def plot_plan(manifest):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = defaultdict(list)
    with (manifest.parent / "summary.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["task"].startswith("mmlu_"):
                continue
            groups[row["task"], row["metric"], row["ruler_length"]].append(
                (int(row["context"]), float(row["value"])))
    output = manifest.parent / "plots"
    output.mkdir(exist_ok=True)
    for (task, metric, length), points in groups.items():
        points.sort()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Model context limit (tokens)")
        ax.set_ylabel(metric)
        ax.set_title(f"{manifest.parent.parent.name} | {task} | input window {length or 'task default'}")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        name = f"{task}_{metric.replace(',', '_')}_{length or 'default'}"
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{name}.{extension}")
        plt.close(fig)


def collect(root):
    plan = json.loads((root / "plan.json").read_text())
    complete = True
    fields = ("size", "mode", "step", "suite", "context", "task", "ruler_length", "metric", "value")
    with (root / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for job in plan["jobs"]:
            directory = Path(job["environment"]["EVAL_OUTPUT"])
            complete &= (directory / "DONE").exists()
            summary = directory / "summary.csv"
            if summary.exists():
                with summary.open() as results:
                    for row in csv.DictReader(results):
                        writer.writerow({**{k: job[k] for k in ("size", "mode", "step", "suite")},
                                         **row})
    if complete:
        (root / "DONE").write_text(f"All {plan['total_shards']} LongRoPE evaluation shards complete.\n")
    print(f"{'COMPLETE' if complete else 'PARTIAL'}: {root / 'summary.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    collect(parser.parse_args().root)
