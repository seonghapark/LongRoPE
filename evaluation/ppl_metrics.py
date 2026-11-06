"""Corpus token perplexity, weighted by the number of scored target tokens."""

import math


def process_results(doc, results):
    (loglikelihood,) = results
    tokens = doc["token_count"]
    if not math.isfinite(loglikelihood) or loglikelihood > 0 or tokens <= 0:
        raise ValueError(f"Invalid rolling likelihood/count: {loglikelihood}, {tokens}")
    return {"token_perplexity": (loglikelihood, tokens)}


def token_perplexity(items):
    items = list(items)
    if not items:
        raise ValueError("No perplexity observations")
    for loglikelihood, tokens in items:
        if not math.isfinite(loglikelihood) or loglikelihood > 0 or tokens <= 0:
            raise ValueError(f"Invalid rolling likelihood/count: {loglikelihood}, {tokens}")
    return math.exp(-math.fsum(item[0] for item in items) / sum(item[1] for item in items))
