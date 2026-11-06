"""Load only a complete, validated ProLong text conversion, never a PG19 fallback."""

import json
import os
from pathlib import Path

SPLIT_POLICY = {
    "version": 1, "key": "sha256(decoded UTF-8 text)",
    "heldout_rule": "int(sha256, 16) % 100 == 0",
    "heldout_fraction": 0.01,
}
SAMPLING_POLICY = {
    "version": 2, "unit": "document", "replacement": False,
    "initial_seed": "SHA256(stage_id + UTC first-training-start timestamp), first 32 bits",
    "epoch_shuffle_seed": "stage_initial_seed + epoch",
    "scope": "all training documents across all 12 subsets",
}


def seed_from_environment():
    value = os.environ.get("AGPT_DATA_SEED")
    if value is None:
        raise ValueError("ProLong requires the persisted stage seed in AGPT_DATA_SEED")
    seed = int(value)
    if not 0 <= seed < 2**32:
        raise ValueError("AGPT_DATA_SEED must be an unsigned 32-bit integer")
    return seed


def stage_seed(run):
    import datetime
    import fcntl
    import hashlib

    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    path = run / "SAMPLING.json"
    with (run / "sampling.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            record = json.loads(path.read_text())
        else:
            if any(run.glob("*/step-*/.metadata")):
                raise ValueError("Cannot generate a new sampling seed for an existing checkpoint")
            started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="microseconds")
            seed = int(hashlib.sha256(f"{run.name}|{started}".encode("ascii")).hexdigest()[:8], 16)
            record = {"policy": SAMPLING_POLICY, "stage": run.name,
                      "started_at_utc": started, "seed": seed}
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, indent=2) + "\n")
            temporary.replace(path)
        if (record["policy"] != SAMPLING_POLICY or record["stage"] != run.name
                or type(record["seed"]) is not int or not 0 <= record["seed"] < 2**32):
            raise ValueError(f"Incompatible persisted sampling seed: {path}")
        print(f"ProLong sampling: stage={run.name} seed={record['seed']} "
              f"started_at_utc={record['started_at_utc']}", flush=True)
        return record["seed"]


def document_key(text):
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_heldout(key):
    return int(key, 16) % 100 == 0


def file_sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_data(path, model=None, size=None, verify_hashes=False):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if (manifest["repo"] != "princeton-nlp/prolong-data-512K" or manifest["split"] != "train"
            or manifest["complete"] is not True or len(manifest["subsets"]) != 12):
        raise ValueError("A full 12-subset ProLong-512K training conversion is required")
    if manifest.get("partition") != SPLIT_POLICY:
        raise ValueError("ProLong conversion must exclude the fixed held-out partition")
    heldout = manifest.get("heldout", {})
    if heldout.get("documents", 0) <= 0 or not heldout.get("files"):
        raise ValueError("A nonempty ProLong held-out partition is required")
    if model is not None:
        digest = file_sha256(Path(model) / "tokenizer.json")
        if manifest["target_tokenizer_sha256"][size] != digest:
            raise ValueError("Training tokenizer differs from the ProLong conversion")
    for stats in [*manifest["subsets"].values(), heldout]:
        if stats["documents"] <= 0 or not stats["files"]:
            raise ValueError("Empty converted subset")
        for item in stats["files"]:
            file = (path / item["path"]).resolve()
            file.relative_to(path.resolve())
            if not file.is_file():
                raise FileNotFoundError(file)
            if verify_hashes:
                if file_sha256(file) != item["sha256"]:
                    raise ValueError(f"Converted corpus checksum mismatch: {file}")
    return manifest


def prepare_ppl_corpus(root, original, corpus, metrics):
    """Materialize the same held-out texts for either runtime's tokenizer."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    root, original, corpus = map(Path, (root, original, corpus))
    fingerprint = file_sha256(original / "tokenizer.json")
    manifest = validate_data(corpus, verify_hashes=True)
    if fingerprint not in manifest["target_tokenizer_sha256"].values():
        raise ValueError("PPL tokenizer differs from the ProLong conversion")
    tokenizer = AutoTokenizer.from_pretrained(str(original), local_files_only=True)
    data, tasks = root / "data", root / "tasks"
    data.mkdir()
    tasks.mkdir()
    path = data / "prolong_ppl.parquet"
    seen = set()
    tokens = longest = long_documents = 0
    schema = pa.schema([("document_id", pa.string()), ("text", pa.string()),
                        ("token_count", pa.int64())])
    with pq.ParquetWriter(path, schema, compression="zstd") as writer:
        for item in manifest["heldout"]["files"]:
            for batch in pq.ParquetFile(corpus / item["path"]).iter_batches(batch_size=16):
                rows = []
                for row in batch.to_pylist():
                    key = document_key(row["text"])
                    if key != row["document_id"] or not is_heldout(key) or key in seen:
                        raise ValueError("Invalid, duplicate or non-held-out ProLong PPL document")
                    count = len(tokenizer.encode(row["text"], add_special_tokens=False))
                    if count <= 0:
                        raise ValueError("Empty ProLong PPL document")
                    rows.append({"document_id": key, "text": row["text"], "token_count": count})
                    seen.add(key)
                    tokens += count
                    longest = max(longest, count)
                    long_documents += count >= 131072
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    if len(seen) != manifest["heldout"]["documents"]:
        raise ValueError("ProLong held-out document count differs from its manifest")
    (tasks / "ppl_metrics.py").symlink_to(Path(metrics).resolve())
    (tasks / "prolong_ppl.yaml").write_text(
        "task: prolong_ppl\ndataset_path: parquet\n"
        f"dataset_kwargs:\n  data_files:\n    test: {json.dumps(str(path))}\n"
        "test_split: test\noutput_type: loglikelihood_rolling\n"
        'doc_to_text: ""\ndoc_to_target: "{{text}}"\n'
        "process_results: !function ppl_metrics.process_results\n"
        "metric_list:\n  - metric: token_perplexity\n"
        "    aggregation: !function ppl_metrics.token_perplexity\n"
        "    higher_is_better: false\nmetadata:\n  version: 1.0\n")
    return {"prolong_ppl": {
        "documents": len(seen), "tokens": tokens,
        "max_document_tokens": longest,
        "documents_at_least_128k": long_documents,
        "split": "fixed held-out from ProLong train; excluded from LongRoPE training",
        "source": str(corpus), "repo": manifest["repo"], "revision": manifest["revision"],
        "partition": SPLIT_POLICY, "tokenizer_sha256": fingerprint,
        "manifest_sha256": file_sha256(corpus / "manifest.json"),
        "sha256": file_sha256(path),
    }}


def load_text(path):
    from datasets import load_dataset

    seed = seed_from_environment()
    path = Path(path)
    manifest = validate_data(path)
    files = [str(path / item["path"]) for stats in manifest["subsets"].values()
             for item in stats["files"]]
    dataset = load_dataset("parquet", data_files={"train": files}, split="train",
                           keep_in_memory=False)
    return dataset.shuffle(seed=seed, keep_in_memory=False)
