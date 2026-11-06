"""Convert local ProLong-512K MDS to document text; never reuse Llama-3 IDs as aGPT IDs."""

import argparse
import hashlib
import json
from pathlib import Path

from runtime.prolong_data import SPLIT_POLICY, document_key, is_heldout

HOME = Path(__file__).resolve().parent
REPO = "princeton-nlp/prolong-data-512K"
REVISION = "0bb2becdbb958cf29804da01ae615eb838170aa4"
SUBSETS = (
    "thestackv1_concat_by_repo-524288", "thestackv1_concat_by_repo-65536",
    "book-524288", "book-65536", "fineweb-edu", "fineweb-2023-50",
    "stackexchange", "dolmawiki", "tuluv2", "arxiv", "openwebmath", "textbooks",
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def documents(row, tokenizer):
    ids = row["input_ids"]
    vocabulary_size = len(tokenizer)
    previous = 0
    for pair in row["indices"]:
        if len(pair) != 2:
            raise ValueError("ProLong indices must contain (start, exclusive_end) pairs")
        start, end = map(int, pair)
        if start != previous or end <= start or end > len(ids):
            raise ValueError(f"Gapped/overlapping/invalid document boundaries: {start}, {end}")
        values = [int(token) for token in ids[start:end]]
        if any(token < 0 or token >= vocabulary_size for token in values):
            raise ValueError("ProLong token ID is outside the source tokenizer vocabulary")
        text = tokenizer.decode(values, skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)
        if not text.strip():
            raise ValueError("Empty decoded ProLong document")
        yield text
        previous = end
    if previous != len(ids):
        raise ValueError("ProLong indices do not cover the packed record")


def convert(source, source_tokenizer, models, output, max_records=None, dataset_factory=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    if dataset_factory is None:
        from streaming import LocalDataset

        dataset_factory = LocalDataset
    output = output.resolve()
    output.relative_to(HOME)
    if max_records is not None and max_records < 1:
        raise ValueError("max_records must be positive")
    tokenizer = AutoTokenizer.from_pretrained(str(source_tokenizer), local_files_only=True)
    if (tokenizer.bos_token_id != 128000 or tokenizer.eos_token_id not in (128001, 128009)
            or len(tokenizer) != 128256):
        raise ValueError("Expected the ProLong/Llama-3 128256-token source tokenizer")
    targets = {size: AutoTokenizer.from_pretrained(str(model), local_files_only=True)
               for size, model in models.items()}
    provenance = json.loads((source / "SOURCE.json").read_text())
    if (provenance["repo"] != REPO or provenance["revision"] != REVISION
            or provenance["metadata_only"] is not False):
        raise ValueError("A full pinned download_data.py snapshot is required")
    pending = output.with_name(output.name + ".pending")
    if output.exists() or pending.exists():
        raise FileExistsError(f"Refusing to overwrite conversion: {output} or {pending}")
    pending.mkdir(parents=True)
    manifest = {
        "repo": REPO, "revision": REVISION, "split": "train",
        "complete": max_records is None,
        "source_tokenizer_sha256": digest(source_tokenizer / "tokenizer.json"),
        "target_tokenizer_sha256": {size: digest(model / "tokenizer.json")
                                    for size, model in models.items()},
        "partition": SPLIT_POLICY,
        "heldout": {"documents": 0, "tokens": {size: 0 for size in models}, "files": []},
        "subsets": {}, "conversion": "decode each MDS indices slice, strip source special tokens; "
                                    "aGPT tokenization occurs at training load",
    }
    heldout_directory = pending / "heldout"
    heldout_directory.mkdir()
    heldout_batch = []
    heldout_seen = set()

    def flush_heldout():
        stats = manifest["heldout"]
        path = heldout_directory / f"test-{len(stats['files']):05d}.parquet"
        pq.write_table(pa.Table.from_pylist(heldout_batch), path, compression="zstd")
        stats["files"].append({"path": f"heldout/{path.name}", "sha256": digest(path)})
        heldout_batch.clear()

    for subset in SUBSETS:
        local = source / subset
        index = local / "index.json"
        metadata = json.loads(index.read_text())
        for shard in metadata["shards"]:
            raw = local / shard["raw_data"]["basename"]
            if not raw.is_file() or raw.stat().st_size != shard["raw_data"]["bytes"]:
                raise ValueError(f"Missing/incomplete local MDS shard: {raw}")
        dataset = dataset_factory(local=str(local))
        directory = pending / subset
        directory.mkdir()
        stats = {"records": 0, "documents": 0, "tokens": {size: 0 for size in models},
                 "index_sha256": digest(index), "files": []}
        batch = []

        def flush():
            path = directory / f"train-{len(stats['files']):05d}.parquet"
            pq.write_table(pa.Table.from_pylist(batch), path, compression="zstd")
            stats["files"].append({"path": f"{subset}/{path.name}", "sha256": digest(path)})
            batch.clear()

        count = len(dataset) if max_records is None else min(len(dataset), max_records)
        for record in range(count):
            for document, text in enumerate(documents(dataset[record], tokenizer)):
                key = document_key(text)
                heldout = is_heldout(key)
                if heldout and key in heldout_seen:
                    continue
                counts = {size: len(target.encode(text, add_special_tokens=False))
                          for size, target in targets.items()}
                if any(value <= 0 for value in counts.values()):
                    raise ValueError("Empty aGPT-tokenized document")
                row = {"text": text, "document_id": key, "subset": subset,
                       "record": record, "document": document}
                destination = manifest["heldout"] if heldout else stats
                destination["documents"] += 1
                for size, value in counts.items():
                    destination["tokens"][size] += value
                if heldout:
                    heldout_seen.add(key)
                    heldout_batch.append(row)
                    if len(heldout_batch) == 64:
                        flush_heldout()
                else:
                    batch.append(row)
                    if len(batch) == 64:
                        flush()
            stats["records"] += 1
        if batch:
            flush()
        if not stats["documents"]:
            raise ValueError(f"No documents in {subset}")
        manifest["subsets"][subset] = stats
    if heldout_batch:
        flush_heldout()
    if manifest["complete"] and not manifest["heldout"]["documents"]:
        raise ValueError("Full conversion has no held-out documents")
    (pending / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    pending.rename(output)
    print(f"Converted {'FULL' if manifest['complete'] else 'SMOKE ONLY'} corpus: {output}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Local snapshot of all 12 MDS subsets")
    parser.add_argument("--source-tokenizer", type=Path, required=True, help="Local Llama-3 tokenizer")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-records", type=int, help="Per subset; smoke-only and rejected for training")
    args = parser.parse_args()
    import sys

    sys.path.insert(0, str(HOME / "runtime"))
    import post50

    models = {size: post50.source(HOME, (size, "independent", 131072))
              for size in ("2b", "20b")}
    convert(args.source, args.source_tokenizer, models, args.output, args.max_records)


if __name__ == "__main__":
    main()
