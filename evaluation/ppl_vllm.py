"""Commit rolling likelihoods per document before an hourly allocation expires."""

from lm_eval.api.registry import register_model
from lm_eval.models.vllm_causallms import VLLM


@register_model("ppl_vllm")
class PerplexityVLLM(VLLM):
    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        results = []
        for request in requests:
            results.extend(super().loglikelihood_rolling([request], disable_tqdm=disable_tqdm))
        return results
