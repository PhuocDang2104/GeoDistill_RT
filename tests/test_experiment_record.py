import csv
import tempfile
import unittest
from pathlib import Path

import torch

from src.experiment_record import build_protocol, collect_training_log, split_fingerprint
from src.metrics import GlobalDepthMetricAccumulator, average_metric_dict, depth_metrics_torch
from src.train_student import append_csv


class GlobalMetricAggregationTest(unittest.TestCase):
    def test_global_rmse_weights_pixels_not_images(self) -> None:
        accumulator = GlobalDepthMetricAccumulator()
        pred_a = torch.tensor([[[[2.0]]]])
        gt_a = torch.tensor([[[[1.0]]]])
        pred_b = torch.ones(1, 1, 1, 9)
        gt_b = torch.ones_like(pred_b)
        accumulator.update(pred_a, gt_a)
        accumulator.update(pred_b, gt_b)

        global_rmse = accumulator.compute()["rmse"]
        macro_rmse = average_metric_dict(
            [depth_metrics_torch(pred_a, gt_a), depth_metrics_torch(pred_b, gt_b)]
        )["rmse"]

        self.assertAlmostEqual(global_rmse, (1.0 / 10.0) ** 0.5, places=6)
        self.assertAlmostEqual(macro_rmse, 0.5, places=6)


class SplitProtocolTest(unittest.TestCase):
    def _write_split(self, root: Path, name: str, sample_ids: list[str]) -> None:
        lines = [f"{sample_id} rgb/{sample_id}.png sparse/{sample_id}.png none K/{sample_id}.txt" for sample_id in sample_ids]
        (root / name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_split_fingerprint_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_split(root, "train.txt", ["drive_a_sync_image_0001", "drive_b_sync_image_0002"])
            first = split_fingerprint(root / "train.txt")
            second = split_fingerprint(root / "train.txt")
            self.assertEqual(first["file_sha256"], second["file_sha256"])
            self.assertEqual(first["ordered_sample_ids_sha256"], second["ordered_sample_ids_sha256"])

    def test_protocol_rejects_train_val_raw_drive_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_split(root, "train.txt", ["drive_a_sync_image_0001"])
            self._write_split(root, "val.txt", ["drive_a_sync_image_0002"])
            self._write_split(root, "test.txt", ["anonymous_0001"])
            paths = {
                "split_root": str(root),
                "train_split": "train.txt",
                "val_split": "val.txt",
                "test_split": "test.txt",
            }
            with self.assertRaisesRegex(ValueError, "Split leakage"):
                build_protocol({"data": {"image_size": [352, 1216]}}, paths)

    def test_training_log_collects_full_history_and_best_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_log.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["epoch", "val_rmse", "epoch_total_seconds"])
                writer.writeheader()
                writer.writerow({"epoch": 0, "val_rmse": 2.0, "epoch_total_seconds": 10.0})
                writer.writerow({"epoch": 1, "val_rmse": 1.5, "epoch_total_seconds": 11.0})
            result = collect_training_log(path)
            self.assertEqual(result["observed_epochs"], 2)
            self.assertEqual(result["best_epoch"], 1)
            self.assertAlmostEqual(result["epoch_time_seconds_total"], 21.0)

    def test_append_csv_preserves_history_when_schema_grows_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_log.csv"
            append_csv(path, {"epoch": 0, "val_rmse": 2.0})
            append_csv(path, {"epoch": 1, "val_rmse": 1.5, "epoch_total_seconds": 11.0})
            with path.open("r", newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["epoch_total_seconds"], "")
            self.assertEqual(rows[1]["epoch_total_seconds"], "11.0")


if __name__ == "__main__":
    unittest.main()
