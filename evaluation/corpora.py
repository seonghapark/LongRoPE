"""Prepare fixed ProLong held-out token-PPL with each model's own tokenizer."""

import sys
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
sys.path.append(str(HOME / "runtime"))
from prolong_data import prepare_ppl_corpus  # noqa: E402


def prepare_corpora(root, original, corpus):
    return prepare_ppl_corpus(root, original, corpus, Path(__file__).with_name("ppl_metrics.py"))
