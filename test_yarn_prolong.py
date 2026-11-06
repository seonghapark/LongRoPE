import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yarn_post50 as yarn
from runtime.prolong_data import SAMPLING_POLICY


class YarnProlongTests(unittest.TestCase):
    def test_prepare_is_isolated_full_sweep_without_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "yarn"
            parent = Path(directory) / "longrope"
            with patch.object(yarn.subprocess, "run") as submit:
                yarn.prepare(root, parent)
                submit.assert_not_called()
            plan = json.loads((root / "plan.json").read_text())
            self.assertEqual(len(plan["stages"]), 14)
            self.assertEqual(plan["exports"], 112)
            self.assertEqual(plan["sampling"], SAMPLING_POLICY)
            self.assertTrue((root / "data/prolong").is_symlink())
            self.assertFalse((root / "GPU_APPROVED.json").exists())
            self.assertFalse((root / "state.json").exists())

    def test_submission_requires_longrope_done_and_gpu_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            parent = home / "experiments/longrope"
            parent.mkdir(parents=True)
            root = home / "experiments/yarn"
            yarn.prepare(root, parent)
            with patch.object(yarn, "HOME", home), patch.object(yarn.subprocess, "run") as qsub:
                with self.assertRaisesRegex(RuntimeError, "has not completed"):
                    yarn.submit(root, {"index": 0, "action": "train"})
                (parent / "DONE").touch()
                with self.assertRaises(FileNotFoundError):
                    yarn.submit(root, {"index": 0, "action": "train"})
                for size in ("2b", "20b"):
                    for step in yarn.core.MILESTONES:
                        path = parent / "models" / f"{size}_independent_131072" / f"step-{step}"
                        path.mkdir(parents=True)
                        (path / "VERIFIED.json").write_text("{}")
                with self.assertRaises(PermissionError):
                    yarn.submit(root, {"index": 0, "action": "train"})
                qsub.assert_not_called()

    def test_parameters_preserve_yarn_parallelism_and_use_prolong(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for size in ("2b", "20b"):
                with patch.object(yarn, "validate_data", return_value={
                        "subsets": {"example": {"tokens": {size: 123456}}}}):
                    params = yarn.parameters(root, (size, "independent", 131072))
                self.assertEqual(params["AGPT_DATASET_NAME"], "prolong")
                self.assertEqual(params["AGPT_TRAIN_TOKENS"], "123456")
                self.assertEqual(params["AGPT_SEQ_LEN"], "131072")
                self.assertEqual(params["AGPT_CP"], "4")
                self.assertEqual(params["AGPT_SKIP_NONFINITE"], "0")
                self.assertEqual(params["AGPT_NAN_RETRY_ALWAYS"], "0")
                self.assertNotIn("LONGROPE_PROFILE", params)
                record = json.loads(
                    (root / "runs" / f"{size}_independent_131072/SAMPLING.json").read_text())
                self.assertEqual(int(params["AGPT_DATA_SEED"]), record["seed"])


if __name__ == "__main__":
    unittest.main()
