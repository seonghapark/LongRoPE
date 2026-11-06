import json
import copy
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HOME = Path(__file__).resolve().parent
sys.path[:0] = [str(HOME / "runtime"),
               str(HOME / "runtime/xpu_torchtitan/torchtitan_repo")]
import prepare_data  # noqa: E402
import search  # noqa: E402
from longrope_profile import validate  # noqa: E402
from prolong_data import (  # noqa: E402
    document_key, is_heldout, load_text, prepare_ppl_corpus, seed_from_environment,
    stage_seed, validate_data,
)


class RetainedTests(unittest.TestCase):
    def test_controller_cannot_submit_without_full_training_data(self):
        import post50

        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            root = home / "experiments/prolong128k"
            root.mkdir(parents=True)
            (root / "plan.json").write_text(json.dumps({"stages": post50.STAGES}))
            with patch.object(post50, "HOME", home), \
                    patch.dict(os.environ, {"LONGROPE_ROOT": str(root)}), \
                    patch.object(sys, "argv", ["post50.py", "start"]), \
                    patch.object(post50.subprocess, "run") as submit:
                with self.assertRaises(FileNotFoundError):
                    post50.main()
                submit.assert_not_called()
                self.assertFalse((root / "state.json").exists())

    def test_retained_prefix_original_amplitude_and_nonprefix_scaled(self):
        import torch
        from transformers import LlamaConfig
        from modeling_longrope import RetainedRotaryEmbedding
        from torchtitan.models.common.longrope import LongRoPE

        for theta in (50000, 500000):
            for retained in search.RETAINED_CHOICES:
                target = 131072
                factors = [1 + 15 * i / 63 for i in range(64)]
                cfg = LlamaConfig(
                    hidden_size=128, num_attention_heads=1, num_key_value_heads=1,
                    max_position_embeddings=target, longrope_retained_tokens=retained,
                    rope_parameters={
                        "rope_type": "longrope", "rope_theta": theta, "factor": 16,
                        "original_max_position_embeddings": 8192,
                        "short_factor": [1.0] * 64, "long_factor": factors})
                for cap in (4096, target):
                    titan = LongRoPE.Config(
                        dim=128, max_seq_len=cap, theta=theta, original_seq_len=8192,
                        target_length=target, retained_tokens=retained,
                        long_factor=tuple(factors), short_factor=(1.0,) * 64).build()
                    positions = sorted({0, 1, max(0, retained - 1), retained, 819, cap - 1})
                    pos = torch.tensor(positions)
                    model = RetainedRotaryEmbedding(cfg, serving_length=cap)
                    cos, sin = model(torch.zeros(1), pos[None])
                    torch.testing.assert_close(cos[0, :, :64], titan.cache[pos].real)
                    torch.testing.assert_close(sin[0, :, :64], titan.cache[pos].imag)
                    keep = pos < retained
                    if keep.any():
                        self.assertTrue(torch.allclose(titan.cache[pos[keep]].abs(),
                                                       torch.ones_like(titan.cache[pos[keep]].real)))
                    scale = math.sqrt(1 + math.log(16) / math.log(8192))
                    self.assertTrue(torch.allclose(titan.cache[pos[~keep]].abs(),
                                                   torch.full_like(titan.cache[pos[~keep]].real, scale)))
                    values = torch.randn(1, cap if cap == 4096 else 512, 1, 128,
                                         requires_grad=True)
                    q, k = titan(values, values)
                    (q.square().mean() + k.square().mean()).backward()
                    self.assertTrue(torch.isfinite(values.grad).all())

    def test_joint_search_covers_prefix_choices_and_preserves_elites(self):
        from test_longrope import original

        with tempfile.TemporaryDirectory() as directory:
            p = search.initial_profile(original(Path(directory) / "model"), 131072)
            candidates = search.population(p, 0, 8, 1234)
            self.assertEqual({p["retained_tokens"] for p in candidates}, set(search.RETAINED_CHOICES))
            for i, candidate in enumerate(candidates):
                validate(candidate)
                candidate.update(validation_nll=2 + i, score_file=f"score_{i}")
            parents = candidates[-2:]
            children = search.population(p, 1, 8, 1234, parents)
            for child, parent in zip(children, parents):
                self.assertEqual(child["retained_tokens"], parent["retained_tokens"])
                self.assertEqual(child["long_factor"], parent["long_factor"])
                self.assertEqual(child["validation_nll"], parent["validation_nll"])

    def test_export_retention_metadata_and_runtime_match(self):
        import score
        import verify
        from test_longrope import original, profile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = original(root / "source")
            p = profile(source, 131072)
            p["retained_tokens"] = 64
            score.model_view(source, root / "export", p)
            cfg = json.loads((root / "export/config.json").read_text())
            self.assertEqual(cfg["longrope_retained_tokens"], 64)
            self.assertTrue(cfg["longrope_requires_custom_runtime"])
            self.assertIn("AutoModelForCausalLM", cfg["auto_map"])
            verify.verify(root / "export")

    def test_hf_custom_auto_model_cached_logits_match_full(self):
        import torch
        from transformers import LlamaConfig, AutoModelForCausalLM
        from modeling_longrope import RetainedLlamaForCausalLM

        cfg = LlamaConfig(
            hidden_size=128, intermediate_size=256, num_hidden_layers=1,
            num_attention_heads=1, num_key_value_heads=1, vocab_size=32,
            max_position_embeddings=131072, longrope_retained_tokens=4,
            rope_parameters={"rope_type": "longrope", "rope_theta": 50000, "factor": 16,
                             "original_max_position_embeddings": 8192,
                             "short_factor": [1.0] * 64, "long_factor": [16.0] * 64})
        cfg.auto_map = {"AutoModelForCausalLM": "modeling_longrope.RetainedLlamaForCausalLM"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            torch.manual_seed(1234)
            model = RetainedLlamaForCausalLM(cfg).eval()
            model.save_pretrained(path)
            (path / "modeling_longrope.py").write_text(
                (HOME / "runtime/modeling_longrope.py").read_text())
            loaded = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True).eval()
            ids = torch.arange(8)[None]
            with torch.no_grad():
                full = loaded(ids).logits[:, -1]
                prefix = loaded(ids[:, :-1], use_cache=True)
                last = loaded(ids[:, -1:], past_key_values=prefix.past_key_values).logits[:, -1]
            torch.testing.assert_close(full, last, atol=1e-5, rtol=1e-5)
            self.assertEqual(loaded.model.rotary_emb.retained, 4)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"AGPT_DATA_SEED": "7777"}))

    class SourceTokenizer:
        bos_token_id = 128000
        eos_token_id = 128001

        def __len__(self):
            return 128256

        def decode(self, tokens, **kwargs):
            return " ".join(f"token{token}" for token in tokens)

        def encode(self, text, **kwargs):
            return list(range(len(text.split())))

    def test_indices_are_preserved_and_invalid_data_fails_explicitly(self):
        tokenizer = self.SourceTokenizer()
        row = {"input_ids": [1, 2, 3, 4], "indices": [(0, 2), (2, 4)]}
        self.assertEqual(list(prepare_data.documents(row, tokenizer)),
                         ["token1 token2", "token3 token4"])
        for indices in ([(0, 3), (2, 4)], [(1, 4)], [(0, 2)], [(0, 9)], [(0, 0)]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                list(prepare_data.documents(dict(row, indices=indices), tokenizer))
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            list(prepare_data.documents({"input_ids": [128256], "indices": [(0, 1)]}, tokenizer))

    def test_vocabulary_size_is_not_recomputed_per_token(self):
        class CountingTokenizer(self.SourceTokenizer):
            calls = 0

            def __len__(self):
                self.calls += 1
                return super().__len__()

        tokenizer = CountingTokenizer()
        row = {"input_ids": [1] * 524288, "indices": [(0, 524288)]}
        self.assertEqual(len(list(prepare_data.documents(row, tokenizer))), 1)
        self.assertEqual(tokenizer.calls, 1)

    def test_real_parquet_conversion_manifest_mixture_and_smoke_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "mds"
            source.mkdir()
            (source / "SOURCE.json").write_text(json.dumps({
                "repo": prepare_data.REPO, "revision": prepare_data.REVISION,
                "metadata_only": False}))
            tokenizer = root / "tokenizer"
            tokenizer.mkdir()
            (tokenizer / "tokenizer.json").write_text("{}")
            for subset in prepare_data.SUBSETS:
                local = source / subset
                local.mkdir()
                (local / "index.json").write_text('{"shards": []}')
            heldout_tokens = [i for i in range(1000, 10000)
                              if is_heldout(document_key(f"token{i} token{i}"))][:20]
            ids = [1, 2, 3, 4] + [token for i in heldout_tokens for token in (i, i)]
            rows = [{"input_ids": ids,
                     "indices": [(i, i + 2) for i in range(0, len(ids), 2)]}]
            with patch.object(prepare_data, "HOME", root), \
                    patch("transformers.AutoTokenizer.from_pretrained",
                          return_value=self.SourceTokenizer()):
                manifest = prepare_data.convert(
                    source, tokenizer, {"2b": tokenizer, "20b": tokenizer},
                    root / "converted", dataset_factory=lambda **kwargs: rows)
                validate_data(root / "converted", model=tokenizer, size="2b", verify_hashes=True)
                self.assertEqual(len(manifest["subsets"]), 12)
                self.assertEqual(sum(s["tokens"]["2b"] for s in manifest["subsets"].values()), 48)
                self.assertEqual(manifest["heldout"]["documents"], 20)
                self.assertEqual(manifest["heldout"]["tokens"]["2b"], 40)
                import pyarrow.parquet as pq

                heldout_ids = set()
                for item in manifest["heldout"]["files"]:
                    for row in pq.read_table(root / "converted" / item["path"]).to_pylist():
                        self.assertTrue(is_heldout(row["document_id"]))
                        heldout_ids.add(row["document_id"])
                for stats in manifest["subsets"].values():
                    for item in stats["files"]:
                        for row in pq.read_table(root / "converted" / item["path"]).to_pylist():
                            self.assertFalse(is_heldout(row["document_id"]))
                            self.assertNotIn(row["document_id"], heldout_ids)
                evaluation = root / "evaluation"
                evaluation.mkdir()
                result = prepare_ppl_corpus(
                    evaluation, tokenizer, root / "converted",
                    HOME / "evaluation/ppl_metrics.py")
                self.assertEqual(result["prolong_ppl"]["documents"], 20)
                self.assertEqual(result["prolong_ppl"]["tokens"], 40)
                evaluated = pq.read_table(evaluation / "data/prolong_ppl.parquet").to_pylist()
                self.assertEqual({row["document_id"] for row in evaluated}, heldout_ids)
                self.assertTrue((evaluation / "tasks/prolong_ppl.yaml").is_file())
                old_manifest = dict(manifest)
                old_manifest.pop("partition")
                (root / "converted/manifest.json").write_text(json.dumps(old_manifest))
                with self.assertRaisesRegex(ValueError, "held-out"):
                    validate_data(root / "converted")
                (root / "converted/manifest.json").write_text(json.dumps(manifest))
                sample = next(iter(load_text(root / "converted")))
                self.assertIn(sample["text"], ("token1 token2", "token3 token4"))
                shuffled = list(load_text(root / "converted"))
                self.assertEqual(shuffled, list(load_text(root / "converted")))
                with patch.dict(os.environ, {"AGPT_DATA_SEED": "8888"}):
                    other_order = list(load_text(root / "converted"))
                self.assertNotEqual(shuffled, other_order)
                self.assertEqual(
                    {(row["subset"], row["record"], row["document"]) for row in shuffled},
                    {(row["subset"], row["record"], row["document"]) for row in other_order})
                self.assertEqual(len(shuffled), 24)
                self.assertEqual(len({(row["subset"], row["record"], row["document"])
                                      for row in shuffled}), 24)
                smoke = prepare_data.convert(
                    source, tokenizer, {"2b": tokenizer, "20b": tokenizer},
                    root / "smoke", max_records=1, dataset_factory=lambda **kwargs: rows)
                self.assertFalse(smoke["complete"])
                with self.assertRaisesRegex(ValueError, "full"):
                    validate_data(root / "smoke")
                (tokenizer / "tokenizer.json").write_text('{"changed": true}')
                with self.assertRaisesRegex(ValueError, "tokenizer"):
                    validate_data(root / "converted", model=tokenizer, size="2b")
                first = next(iter(manifest["subsets"].values()))["files"][0]
                (root / "converted" / first["path"]).write_text("corrupted")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    validate_data(root / "converted", verify_hashes=True)

    def test_partition_is_content_based_and_stable(self):
        training = heldout = 0
        for index in range(10000):
            key = document_key(f"document {index}")
            self.assertEqual(key, prepare_data.document_key(f"document {index}"))
            if is_heldout(key):
                heldout += 1
            else:
                training += 1
        self.assertEqual(training + heldout, 10000)
        self.assertGreater(heldout, 50)
        self.assertLess(heldout, 150)

    def test_prolong_epoch_shuffle_and_mid_epoch_resume(self):
        from datasets import Dataset
        from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataset

        source = Dataset.from_list([{"text": str(i)} for i in range(100)]).shuffle(seed=7777)

        class Tokenizer:
            def encode(self, text, **kwargs):
                return [int(text)] * 3

        with patch("torchtitan.hf_datasets.text_datasets._validate_dataset",
                   return_value=("unused", lambda path: source, lambda row: row["text"])):
            dataset = HuggingFaceTextDataset("prolong", None, Tokenizer(),
                                            seq_len=2, infinite=True)
            first = [row["text"] for row in dataset._get_data_iter()]
            dataset.reloop()
            second = [row["text"] for row in dataset._get_data_iter()]
            self.assertEqual(set(first), set(second))
            self.assertNotEqual(first, second)
            self.assertEqual(second, source.shuffle(seed=7778)["text"])
            iterator = iter(dataset)
            for _ in range(7):
                next(iterator)
            saved = copy.deepcopy(dataset.state_dict())
            resumed = HuggingFaceTextDataset("prolong", None, Tokenizer(),
                                            seq_len=2, infinite=True)
            resumed.load_state_dict(saved)
            invalid = dict(saved, shuffle_seed=8888)
            with self.assertRaisesRegex(ValueError, "seed differs"):
                resumed.load_state_dict(invalid)
            restored = iter(resumed)
            for _ in range(20):
                expected, expected_label = next(iterator)
                actual, actual_label = next(restored)
                self.assertEqual(expected["input"].tolist(), actual["input"].tolist())
                self.assertEqual(expected["positions"].tolist(), actual["positions"].tolist())
                self.assertEqual(expected_label.tolist(), actual_label.tolist())

    def test_initial_stage_seed_depends_on_start_time_and_is_reused(self):
        import hashlib

        with tempfile.TemporaryDirectory() as directory, patch("datetime.datetime") as clock:
            root = Path(directory)
            stamp = "2026-11-05T16:04:39.123456+00:00"
            clock.now.return_value.isoformat.return_value = stamp
            run = root / "first/2b_shared_16384"
            seed = stage_seed(run)
            self.assertEqual(seed, int(hashlib.sha256(
                f"2b_shared_16384|{stamp}".encode("ascii")).hexdigest()[:8], 16))
            clock.now.return_value.isoformat.return_value = "2026-11-06T17:05:40.654321+00:00"
            self.assertEqual(stage_seed(run), seed)
            clock.now.assert_called_once()
            later = stage_seed(root / "second/2b_shared_16384")
            self.assertNotEqual(seed, later)
            record = json.loads((run / "SAMPLING.json").read_text())
            self.assertEqual(record["started_at_utc"], stamp)
            self.assertEqual(record["seed"], seed)
            checkpoint = root / "missing_seed/step-10"
            checkpoint.mkdir(parents=True)
            (checkpoint / ".metadata").touch()
            with self.assertRaisesRegex(ValueError, "existing checkpoint"):
                stage_seed(root)

    def test_seed_environment_is_required_and_validated(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "persisted"):
            seed_from_environment()
        for seed in ("-1", "4294967296", "invalid"):
            with patch.dict(os.environ, {"AGPT_DATA_SEED": seed}), self.assertRaises(ValueError):
                seed_from_environment()


if __name__ == "__main__":
    unittest.main()
