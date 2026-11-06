import importlib.util
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

HOME = Path(__file__).resolve().parent
sys.path[:0] = [str(HOME / "runtime"), str(HOME / "evaluation"),
               str(HOME / "runtime/xpu_torchtitan/torchtitan_repo")]
import post50  # noqa: E402
import search  # noqa: E402
import score  # noqa: E402
import verify  # noqa: E402
from longrope_profile import patch_model, save, scaling, validate  # noqa: E402


def original(path, theta=50000):
    path.mkdir(parents=True)
    save(path / "config.json", {
        "model_type": "llama", "hidden_size": 2048, "num_attention_heads": 16,
        "num_key_value_heads": 4, "num_hidden_layers": 24, "intermediate_size": 8192,
        "max_position_embeddings": 131072, "rope_theta": theta})
    (path / "model.safetensors").touch()
    (path / "tokenizer.json").write_text("{}")
    return path


def profile(source, length):
    p = search.initial_profile(source, length)
    p.update(status="searched", validation_nll=2.0)
    p["long_factor"] = [1 + (length / 8192 - 1) * i / 63 for i in range(64)]
    return p


class LongRoPETests(unittest.TestCase):
    def test_vectors_geometry_source_and_search_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            source = original(Path(directory) / "original")
            p = profile(source, 32768)
            validate(p, dim=128, length=32768, source=source, searched=True)
            self.assertEqual(len(scaling(p)["long_factor"]), 64)
            for changes in ({"long_factor": [1]}, {"short_factor": [float("nan")] * 64},
                            {"retained_tokens": -1}, {"retained_tokens": 8192},
                            {"status": "candidate"},
                            {"validation_nll": float("inf")}, {"theta": -1}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    validate(dict(p, **changes), searched=True)
            with self.assertRaises(ValueError):
                validate(p, length=65536)
            (source / "config.json").write_text("{}")
            with self.assertRaises(ValueError):
                validate(p, source=source)

    def test_view_and_export_never_modify_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = original(root / "original")
            before = (source / "config.json").read_bytes()
            p = profile(source, 16384)
            output = root / "candidate"
            score.model_view(source, output, p)
            self.assertEqual((source / "config.json").read_bytes(), before)
            self.assertFalse((output / "config.json").is_symlink())
            self.assertTrue((output / "model.safetensors").is_symlink())
            cfg = json.loads((output / "config.json").read_text())
            self.assertEqual(cfg["rope_scaling"]["rope_type"], "longrope")
            self.assertNotIn("beta_fast", cfg["rope_scaling"])
            verify.verify(output)

    def test_hf_parity_both_sizes_all_lengths_and_gradients(self):
        import torch
        from torchtitan.models.common.longrope import LongRoPE

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for size, theta in (("2b", 50000), ("20b", 500000)):
                source = original(root / size, theta)
                for length in (16384, 32768, 65536, 131072):
                    output = root / f"{size}_{length}"
                    score.model_view(source, output, profile(source, length))
                    with self.subTest(size=size, length=length):
                        verify.verify(output)
            rope = LongRoPE.Config(
                dim=128, max_seq_len=64, theta=50000, original_seq_len=8192,
                target_length=16384, long_factor=(2.0,) * 64, short_factor=(1.0,) * 64).build()
            self.assertIsInstance(rope, LongRoPE)
            query = torch.randn(1, 64, 2, 128, requires_grad=True)
            key = torch.randn(1, 64, 2, 128, requires_grad=True)
            q, k = rope(query, key)
            (q.square().mean() + k.square().mean()).backward()
            self.assertTrue(torch.isfinite(query.grad).all())
            self.assertTrue(torch.isfinite(key.grad).all())

    def test_agpt_hook_selects_longrope_and_preserves_qk_permutation(self):
        from torchtitan.models.agpt.config_registry import agpt_2b_yarn, agpt_20b_yarn
        from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
        from torchtitan.models.common.longrope import LongRoPE

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, factory, theta in (("2b", agpt_2b_yarn, 50000),
                                         ("20b", agpt_20b_yarn, 500000)):
                source = original(root / name, theta)
                path = root / f"{name}.json"
                save(path, profile(source, 16384))
                training = factory()
                training.training.seq_len = 16384
                cfg = training.model_spec.model
                with patch.dict(os.environ, {"LONGROPE_PROFILE": str(path)}):
                    cfg.update_from_config(config=training)
                self.assertTrue(all(isinstance(layer.attention.rope, LongRoPE.Config)
                                    for layer in cfg.layers))
                adapter = Llama3StateDictAdapter.__new__(Llama3StateDictAdapter)
                adapter.model_config = cfg
                self.assertTrue(adapter._needs_qk_permute())
            with patch.dict(os.environ, {"LONGROPE_PROFILE": str(root / "missing.json")}):
                with self.assertRaises(FileNotFoundError):
                    cfg._set_longrope(16384)

    def test_real_export_adapter_configuration_uses_longrope(self):
        import dcp_to_hf
        from torchtitan.models.common.longrope import LongRoPE

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for size, theta in (("2b", 50000), ("20b", 500000)):
                source = original(root / size, theta)
                path = root / f"{size}.json"
                save(path, profile(source, 32768))
                with patch.dict(os.environ, {"LONGROPE_PROFILE": str(path)}):
                    adapter = dcp_to_hf._build_adapter(
                        ["--module", "agpt", "--config", f"agpt_{size}_yarn",
                         "--training.seq_len", "32768"], str(source))
                self.assertIsInstance(adapter.model_config.layers[0].attention.rope,
                                      LongRoPE.Config)
                self.assertTrue(adapter._needs_qk_permute())

    def test_evolution_is_reproducible_nonuniform_and_elitist(self):
        with tempfile.TemporaryDirectory() as directory:
            source = original(Path(directory) / "original")
            p = profile(source, 131072)
            first = search.population(p, 0, 8, 1234)
            self.assertEqual(first, search.population(p, 0, 8, 1234))
            self.assertGreater(len(set(first[2]["long_factor"])), 1)
            parents = first[:2]
            for parent in parents:
                parent["score_file"] = "fixture/score.json"
            following = search.population(p, 1, 8, 1234, parents)
            self.assertEqual(following[0]["long_factor"], parents[0]["long_factor"])
            for candidate in following:
                validate(candidate)
                self.assertEqual(candidate["long_factor"], sorted(candidate["long_factor"]))
                self.assertTrue(all(1 <= v <= 16 for v in candidate["long_factor"]))

    def test_search_is_a_real_prerequisite_and_training_parameters_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = original(root / "original")
            (root / "profiles").mkdir()
            with patch.object(post50, "source", return_value=source):
                self.assertEqual(post50.next_action(root, 0)["action"], "search")
                with self.assertRaises(FileNotFoundError):
                    post50.parameters(root, post50.STAGES[0])
                save(root / "profiles/2b_independent_131072.json", profile(source, 131072))
                self.assertEqual(post50.next_action(root, 0)["action"], "train")
                with patch("prolong_data.validate_data", return_value={
                        "subsets": {"fixture": {"tokens": {"2b": 100}}}}):
                    env = post50.parameters(root, post50.STAGES[0])
                self.assertEqual(env["AGPT_KEEP_EVERY"], "50")
                self.assertEqual(env["AGPT_CKPT_INTERVAL"], "10")
                self.assertEqual(env["AGPT_NAN_RETRY_ALWAYS"], "0")
                self.assertIn(str(HOME), env["TORCHTITAN_ROOT"])
                self.assertEqual(env["AGPT_DATASET_DIR"], str(root / "data/prolong"))
                self.assertEqual(len(post50.STAGES) * len(post50.MILESTONES), 16)

    def test_resumable_search_finishes_with_source_bound_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            root = home / "experiments/post50"
            (root / "search").mkdir(parents=True)
            (root / "profiles").mkdir()
            source = original(home / "original")

            def prepare_calibration(source, length, path, samples):
                save(path, {"split": "validation", "target_length": length,
                            "tokenizer_sha256": hashlib.sha256(
                                (source / "tokenizer.json").read_bytes()).hexdigest(),
                            "samples": [{"document": i, "tokens": [0] * (length - 2)}
                                        for i in range(samples)]})

            def score_candidate(command, **kwargs):
                candidate = json.loads(Path(command[3]).read_text())
                data = Path(command[4])
                save(Path(command[5]), {"nll": sum(candidate["long_factor"]) / 64,
                                       "scored_tokens": 2 * (131072 - 2),
                                       "calibration_sha256": hashlib.sha256(data.read_bytes()).hexdigest()})

            args = SimpleNamespace(root=root, stage=0, budget_seconds=1000, population=4,
                                   generations=2, samples=2, seed=1234)
            with patch.object(search, "HOME", home), \
                    patch.object(post50, "source", return_value=source), \
                    patch.object(search, "calibration", side_effect=prepare_calibration):
                import subprocess

                with patch.object(search.subprocess, "run",
                                  side_effect=subprocess.TimeoutExpired("score", 10)):
                    self.assertEqual(search.run(args), 124)
                self.assertFalse((root / "profiles/2b_independent_131072.json").exists())
                with patch.object(search.subprocess, "run", side_effect=score_candidate) as scorer:
                    self.assertEqual(search.run(args), 0)
                    self.assertEqual(scorer.call_count, 6)
            winner = json.loads((root / "profiles/2b_independent_131072.json").read_text())
            validate(winner, source=source, searched=True)
            self.assertEqual(winner["validation_nll"], 1.0)
            self.assertTrue(Path(winner["score_file"]).is_file())

    def test_cumulative_parent_is_the_verified_longrope_400_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = post50.model(root, ("20b", "shared", 16384))
            parent.mkdir(parents=True)
            with self.assertRaises(FileNotFoundError):
                post50.source(root, ("20b", "cumulative", 32768))
            save(parent / "VERIFIED.json", {"step": 400})
            self.assertEqual(post50.source(root, ("20b", "cumulative", 32768)), parent)

    def test_full_evaluation_plans_and_no_submission(self):
        spec = importlib.util.spec_from_file_location("longrope_evaluation_prepare",
                                                     HOME / "evaluation/prepare.py")
        prepare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prepare)
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / "experiments/post50"
            (root / "evaluation").mkdir(parents=True)
            base = root / "assets"
            for size, theta, name in (
                    ("2b", 50000, "agpt-2b-v2-256n-step-92859-safetensors"),
                    ("20b", 500000, "agpt-20b-v2-512n-step-4000-safetensors")):
                original(base / name, theta)
            with patch.object(post50, "BASE", base), patch.object(prepare, "BASE", base), \
                    patch.object(prepare, "HOME", workspace):
                with self.assertRaises(FileNotFoundError):
                    prepare.prepare(root)
                for stage in post50.STAGES:
                    source = post50.source(root, stage)
                    for step in post50.MILESTONES:
                        output = post50.model(root, stage, step)
                        output.mkdir(parents=True)
                        (output / "config.json").write_text((source / "config.json").read_text())
                        (output / "tokenizer.json").write_text("{}")
                        (output / "model.safetensors").touch()
                        patch_model(output, profile(source, stage[2]))
                        save(output / "VERIFIED.json", {"step": step})
                import corpora

                with patch.object(corpora, "prepare_corpora", return_value={}), \
                        patch("subprocess.run") as submit:
                    jobs = prepare.prepare(root)
                    submit.assert_not_called()
                self.assertEqual(len(jobs), 54)
                self.assertEqual(sum(j["shards"] for j in jobs), 1746)
                self.assertEqual(sum(j["shards"] for j in jobs if j["suite"] == "ppl"), 108)
                self.assertEqual(sum(j["shards"] for j in jobs if j["suite"] == "mmlu5"), 18)
                self.assertEqual({j["environment"]["TP"] for j in jobs if j["size"] == "20b"}, {"4"})
                self.assertEqual({j["environment"]["EVAL_QUEUE"] for j in jobs}, {"debug"})


if __name__ == "__main__":
    unittest.main()
