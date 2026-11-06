"""Explicit, revision-pinned download of ProLong MDS assets; not called by training."""

import argparse
import json

from prepare_data import HOME, REPO, REVISION, SUBSETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-only", action="store_true",
                        help="Download README and indices, not the ~31B-token corpus")
    args = parser.parse_args()
    from huggingface_hub import HfApi, hf_hub_download

    destination = HOME / "experiments/prolong128k/data/source"
    destination.mkdir(parents=True, exist_ok=True)
    if args.metadata_only:
        files = ["README.md", *[f"{subset}/index.json" for subset in SUBSETS]]
    else:
        files = HfApi().list_repo_files(REPO, repo_type="dataset", revision=REVISION)
    for filename in files:
        hf_hub_download(repo_id=REPO, repo_type="dataset", revision=REVISION,
                        filename=filename, local_dir=destination)
    (destination / "SOURCE.json").write_text(json.dumps({
        "repo": REPO, "revision": REVISION, "metadata_only": args.metadata_only,
    }, indent=2) + "\n")
    print(f"{'Metadata only' if args.metadata_only else 'Full snapshot'}: {destination}")


if __name__ == "__main__":
    main()
