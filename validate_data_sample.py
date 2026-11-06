"""CPU-only check of one real ProLong MDS record and its Llama-3 decoding."""

import json

from prepare_data import HOME, REPO, REVISION, digest, documents


def main():
    print("Loading MDS and tokenizer libraries", flush=True)
    from huggingface_hub import HfApi, hf_hub_download
    from streaming import LocalDataset
    from transformers import AutoTokenizer

    print("Downloading one pinned MDS shard (not the full corpus)", flush=True)
    sample = HOME / "validation/mds_sample"
    local = sample / "raw"
    subset = "book-524288"
    source = HOME / "experiments/prolong128k/data/source" / subset / "index.json"
    index = json.loads(source.read_text())
    first = index["shards"][0]
    hf_hub_download(REPO, repo_type="dataset", revision=REVISION,
                    filename=f"{subset}/{first['raw_data']['basename']}", local_dir=local)
    print("MDS shard ready; preparing the public ProLong tokenizer", flush=True)
    (local / subset / "index.json").write_text(json.dumps({"version": 2, "shards": [first]}))
    tokenizer_repo = "princeton-nlp/Llama-3-8B-ProLong-512k-Base"
    tokenizer_revision = HfApi().model_info(tokenizer_repo).sha
    directory = sample / "tokenizer"
    for filename in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        hf_hub_download(tokenizer_repo, filename=filename, revision=tokenizer_revision,
                        local_dir=directory)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    if len(tokenizer) != 128256 or tokenizer.bos_token_id != 128000:
        raise ValueError("Unexpected upstream ProLong tokenizer")
    dataset = LocalDataset(local=str(local / subset))
    print("Reading the first actual packed MDS record", flush=True)
    row = dataset[0]
    texts = list(documents(row, tokenizer))
    import sys

    sys.path.insert(0, str(HOME / "runtime"))
    import post50

    target_counts = {}
    target_hashes = {}
    for size in ("2b", "20b"):
        model = post50.source(HOME, (size, "independent", 131072))
        target = AutoTokenizer.from_pretrained(model, local_files_only=True)
        total = 0
        vocabulary_size = len(target)
        for text in texts:
            ids = target.encode(text, add_special_tokens=False)
            if not ids or any(token < 0 or token >= vocabulary_size for token in ids):
                raise ValueError(f"Invalid re-tokenized aGPT-{size} IDs")
            total += len(ids)
        target_counts[size] = total
        target_hashes[size] = digest(model / "tokenizer.json")
    report = {
        "repo": REPO, "revision": REVISION, "subset": subset, "record": 0,
        "source_tokens": len(row["input_ids"]), "documents": len(texts),
        "decoded_characters": sum(len(text) for text in texts),
        "tokenizer_repo": tokenizer_repo, "tokenizer_revision": tokenizer_revision,
        "source_tokenizer_sha256": digest(directory / "tokenizer.json"),
        "agpt_tokens": target_counts, "agpt_tokenizer_sha256": target_hashes,
        "full_training_conversion": False, "gpu_execution": False,
    }
    (sample / "RESULT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
