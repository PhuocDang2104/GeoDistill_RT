import copy
import unittest

import torch

from migrate_anchorflow_v8_bf16 import migrate_payload


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.old = {"config": {"amp": "fp16"}, "source_sha256": "same", "data": "same"}
        self.new = {"config": {"amp": "bf16"}, "source_sha256": "same", "data": "same"}
        self.payload = {
            "protocol": copy.deepcopy(self.old), "model": {"weight": torch.ones(3)},
            "optimizer": {"state": {0: {"exp_avg": torch.ones(3), "step": torch.tensor(3.)}}},
            "scaler": {"scale": 4096.}, "epoch": 4, "global_step": 2000,
            "best_rmse": 1.097, "early_stopping": {"bad_epochs": 0},
            "rng_torch": torch.get_rng_state(),
        }

    def test_preserves_training_state_and_original(self):
        recovered = migrate_payload(self.payload, self.old, self.new)
        self.assertEqual(recovered["scaler"], {})
        self.assertEqual(recovered["protocol"], self.new)
        self.assertEqual(self.payload["protocol"], self.old)
        self.assertEqual(self.payload["scaler"], {"scale": 4096.})
        for key in ("model", "optimizer", "early_stopping", "rng_torch"):
            self.assertIs(recovered[key], self.payload[key])
        self.assertEqual(recovered["epoch"], 4)
        self.assertEqual(recovered["global_step"], 2000)

    def test_rejects_protocol_change(self):
        with self.assertRaisesRegex(RuntimeError, "protocol mismatch"):
            migrate_payload(self.payload, {}, self.new)

    def test_rejects_corrupt_model(self):
        self.payload["model"]["weight"][0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "model state"):
            migrate_payload(self.payload, self.old, self.new)

    def test_rejects_corrupt_optimizer(self):
        self.payload["optimizer"]["state"][0]["exp_avg"][0] = float("inf")
        with self.assertRaisesRegex(RuntimeError, "optimizer state"):
            migrate_payload(self.payload, self.old, self.new)


if __name__ == "__main__":
    unittest.main()
