import io
import inspect
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

from data import KITTIDataset, canonical_id, extract_metric, safe_extract, teacher_arrays, write_json
from metrics import Metrics
from losses import objective
from loss_helpers import range_rmse, teacher_weights
from model_v3 import AnchorFlowEdge
from core import PhaseUpsample, divergence
from run import learning_rate


def loss_config(enabled=True):
    import json
    config = json.loads(Path(__file__).with_name("config.json").read_text())
    config.update(teacher_enabled=enabled, kd_conf_min=.1)
    return config


def enrich(pred):
    d = pred["D1"]
    return {**pred, "D_full": d, "sensor_logits": d*0,
            "sensor_gate": d*0, "delta4": d*0, "delta1": d*0}


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_inference_interface_and_forward_backward(self):
        self.assertEqual(list(inspect.signature(AnchorFlowEdge.forward).parameters), ["self", "rgb", "sparse", "mask", "K"])
        model = AnchorFlowEdge(pretrained=False).train()
        self.assertTrue(torch.equal(model.rgb_mean, torch.full((1, 3, 1, 1), 0.5)))
        rgb = torch.rand(2, 3, 64, 128)
        sparse = 4 + torch.rand(2, 1, 64, 128) * 110
        mask = (torch.rand_like(sparse) < 0.1).float()
        K = torch.tensor([[[80., 0, 64], [0, 80, 32], [0, 0, 1]]]).repeat(2, 1, 1)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = model(rgb, sparse, mask, K)
        self.assertEqual(output["D_full"].shape, sparse.shape)
        self.assertTrue(torch.equal(output["D_hard"][mask.bool()], sparse[mask.bool()]))
        self.assertTrue(all(torch.isfinite(v).all() for v in output.values()))
        batch = {"gt": torch.full_like(sparse, 20), "gt_mask": (torch.rand_like(sparse) < 0.15).float(),
                 "sparse": sparse * mask, "mask": mask,
                 "teacher": torch.full_like(sparse, 18), "confidence": torch.full_like(sparse, 0.8)}
        loss, _ = objective(output, batch, torch.zeros_like(mask), 2, loss_config())
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(float(model.decoder.head16.weight.grad.abs().sum()), 0)
        model.eval()
        with torch.no_grad():
            empty = model(rgb, torch.zeros_like(sparse), torch.zeros_like(mask), K)
        self.assertTrue(torch.isfinite(empty["D_full"]).all())
        self.assertGreaterEqual(float(empty["D_full"].min()), 0.09999)

    def test_kd_excludes_gt_and_all_original_sparse(self):
        gt_mask = torch.zeros(1, 1, 16, 16)
        gt_mask[..., 0, 0] = 1
        mask = torch.zeros_like(gt_mask)
        mask[..., 2, 2] = 1
        depth = torch.full_like(gt_mask, 10., requires_grad=True)
        predictions = {name: depth for name in ("D16", "D8", "D4", "D2", "D1")}
        batch = {"gt": torch.ones_like(depth) * 15, "gt_mask": gt_mask, "mask": mask,
                 "sparse": mask * 12, "teacher": torch.ones_like(depth) * 20, "confidence": torch.ones_like(depth)}
        kd_total, _ = objective(enrich(predictions), batch, mask, 2, loss_config(True))
        base_total, _ = objective(enrich(predictions), batch, mask, 2, loss_config(False))
        (kd_total - base_total).backward()
        # Subtracting two complete objectives may leave float32 round-off.
        self.assertAlmostEqual(float(depth.grad[..., 0, 0]), 0, delta=1e-6)
        self.assertAlmostEqual(float(depth.grad[..., 2, 2]), 0, delta=1e-6)
        self.assertLess(float(depth.grad[..., 5, 5]), 0)
        teacher, weights, _, _ = teacher_weights(batch, 0.1)
        self.assertEqual(float(weights[..., 2, 2]), 0)
        self.assertTrue(torch.isfinite(teacher).all())

    def test_confidence_is_not_normalized_away(self):
        d = torch.ones(1, 1, 16, 16) * 10
        pred = {key: d for key in ("D16", "D8", "D4", "D2", "D1")}
        batch = {"gt": torch.zeros_like(d), "gt_mask": torch.zeros_like(d), "sparse": torch.zeros_like(d),
                 "mask": torch.zeros_like(d), "teacher": d * 2, "confidence": torch.ones_like(d)}
        a, _ = objective(enrich(pred), batch, batch["mask"], 2, loss_config())
        batch["confidence"] *= 0.2
        b, _ = objective(enrich(pred), batch, batch["mask"], 2, loss_config())
        self.assertAlmostEqual(float(b / a), 0.2, places=5)
        batch["teacher"].fill_(float("nan"))
        empty, _ = objective(enrich(pred), batch, batch["mask"], 2, loss_config())
        self.assertTrue(torch.isfinite(empty))
        self.assertEqual(float(empty), 0)

    def test_coarse_kd_blocks_entire_gt_or_sensor_cell(self):
        shape = (1, 1, 16, 16)
        mask = torch.zeros(shape)
        gt_mask = mask.clone()
        gt_mask[..., 1, 1] = 1
        mask[..., 7, 7] = 1
        pred = {key: torch.full((1, 1, 16 // scale, 16 // scale), 10., requires_grad=True)
                for key, scale in (("D16", 16), ("D8", 8), ("D4", 4), ("D2", 2), ("D1", 1))}
        batch = {"gt": torch.ones(shape) * 15, "gt_mask": gt_mask, "mask": mask,
                 "sparse": mask * 12, "teacher": torch.ones(shape) * 20, "confidence": torch.ones(shape)}
        kd, _ = objective(enrich(pred), batch, mask, 2, loss_config(True))
        gt, _ = objective(enrich(pred), batch, mask, 2, loss_config(False))
        gradient = torch.autograd.grad(kd, pred["D4"], retain_graph=True)[0] - torch.autograd.grad(gt, pred["D4"])[0]
        self.assertLess(float(gradient[..., 3, 3]), 0)
        self.assertAlmostEqual(float(gradient[..., 0, 0]), 0, delta=1e-7)
        self.assertAlmostEqual(float(gradient[..., 1, 1]), 0, delta=1e-7)

    def test_global_pixel_metrics_and_inverse_units(self):
        metrics = Metrics("cpu")
        pred = torch.tensor([[[[2., 4.]]]])
        gt = torch.tensor([[[[1., 2.]]]])
        metrics.update(pred, gt, torch.ones_like(gt), torch.zeros(1, 3, 1, 2))
        report = metrics.report()["all"]
        self.assertAlmostEqual(report["rmse_m"], (2.5) ** .5)
        self.assertAlmostEqual(report["irmse_km_inv"], ((500 ** 2 + 250 ** 2) / 2) ** .5)
        self.assertEqual(report["pixels"], 2)

    def test_range_term_is_finite_and_ignores_insufficient_support(self):
        target = torch.cat((torch.full((1, 1, 8, 8), 10.), torch.full((1, 1, 8, 8), 70.)), -1)
        prediction = (target + 2).requires_grad_()
        value, active = range_rmse(prediction, target, torch.ones_like(target))
        self.assertEqual(int(active), 2)
        self.assertAlmostEqual(float(value.detach()), 1.999, places=4)
        value.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        empty, active = range_rmse(prediction, target, torch.zeros_like(target))
        self.assertEqual(float(empty.detach()), 0)
        self.assertEqual(int(active), 0)

    def test_stream_cache_contains_training_teachers_only(self):
        ids = [f"2011_09_26_drive_0001_sync_image_{i:010d}_image_02" for i in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "metric.tar"
            with tarfile.open(archive, "w") as handle:
                for sid in ids:
                    stream = io.BytesIO()
                    np.savez(stream, D_cm=np.ones((8, 16), np.float32) * 20, C_cm=np.ones((8, 16), np.float32))
                    raw = stream.getvalue()
                    info = tarfile.TarInfo(f"metric_coarse/train/{sid}.npz")
                    info.size = len(raw)
                    handle.addfile(info, io.BytesIO(raw))
            report = extract_metric(archive, ids[:2], ids[2:], root / "cache", .1)
            self.assertEqual(report["cached_train"], 2)
            self.assertEqual(len(list((root / "cache").glob("*.npy"))), 2)
            self.assertFalse((root / "cache" / f"{ids[2]}.npy").exists())
            self.assertEqual(len(report["selected_teacher_content_sha256"]), 64)

    def test_diffusion_zero_flux_and_stability(self):
        c = torch.randn(2, 1, 12, 16)
        g = torch.rand(2, 2, 12, 16)
        d = divergence(c, g)
        self.assertLess(float(d.sum().abs()), 2e-5)
        self.assertEqual(float(divergence(torch.ones_like(c), g).abs().max()), 0)
        self.assertLessEqual(float((c + 0.2 * d).square().sum()), float(c.square().sum()) + 1e-5)

    def test_phase_constant_preservation(self):
        up = PhaseUpsample(16, 16, 1., 0.05).eval()
        with torch.no_grad():
            depth = up(torch.full((1, 1, 8, 12), 25.), torch.randn(1, 16, 8, 12))
        self.assertEqual(depth.shape, (1, 1, 16, 24))
        torch.testing.assert_close(depth, torch.full_like(depth, 25.), atol=1e-5, rtol=1e-5)

    def test_teacher_loader_schema_and_ids(self):
        new = "2011_09_26_drive_0001_sync_image_0000000038_image_03"
        old = "2011_09_26_drive_0001_sync_image_03_0000000038"
        self.assertEqual(canonical_id(old + ".npz"), new)
        self.assertEqual(canonical_id(new + ".npz"), new)
        with io.BytesIO() as stream:
            np.savez(stream, D_cm=np.ones((4, 8), np.float32) * 20, C_cm=np.ones((4, 8), np.float32) * .7)
            stream.seek(0)
            with np.load(stream) as payload:
                arrays = teacher_arrays(payload, (8, 16))
            self.assertEqual(arrays.shape, (2, 8, 16))
            self.assertTrue(np.allclose(arrays[0], 20))
            self.assertTrue(np.allclose(arrays[1], .7))
        with self.assertRaises(ValueError):
            KITTIDataset({"work": "unused"}, "val", teacher=True)
        with self.assertRaises(ValueError):
            KITTIDataset({"work": "unused"}, "test", teacher=True)

    def test_tar_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            tar = base / "bad.tar"
            with tarfile.open(tar, "w") as archive:
                info = tarfile.TarInfo("../escaped.txt")
                info.size = 1
                archive.addfile(info, io.BytesIO(b"x"))
            with self.assertRaises(RuntimeError):
                safe_extract(tar, base / "out")
            self.assertFalse((base / "escaped.txt").exists())

    def test_lr_schedule_endpoints(self):
        self.assertAlmostEqual(learning_rate(9, 100, 10, .05), 1)
        self.assertAlmostEqual(learning_rate(99, 100, 10, .05), .05)

    def test_train_validate_checkpoint_and_resume_on_cpu_fixture(self):
        self._run_training_fixture("v6_connection")

    def test_v5_matched_control_train_and_resume_on_cpu_fixture(self):
        self._run_training_fixture("v5_piecewise")

    def _run_training_fixture(self, model_name):
        # CPU fixture exercises the actual trainer. It does not claim CUDA validation.
        import json
        import run
        cfg = json.loads(Path(__file__).with_name("config.json").read_text())
        cfg["model_name"] = model_name
        cfg["architecture"] = "AnchorFlow-v5-PiecewiseSurface" if model_name != "v3" else "AnchorFlow-Edge-v3-RefineKD"
        rgb = torch.rand(2, 3, 32, 64)
        mask = (torch.rand(2, 1, 32, 64) < .1).float()
        gt = torch.full_like(mask, 20.)
        batch = {"rgb": rgb, "sparse": gt * mask, "mask": mask, "gt": gt,
                 "gt_mask": (torch.rand_like(mask) < .1).float(),
                 "teacher": gt.clone(), "confidence": torch.full_like(mask, .8),
                 "K": torch.tensor([[[40., 0, 32], [0, 40, 16], [0, 0, 1]]]).repeat(2, 1, 1)}
        class TinyLoader(list):
            dataset = [0, 1]
        loader = TinyLoader([batch])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg.update(work=str(root / "work"), drive_runs=str(root / "drive"), epochs=1,
                       workers=0, encoder_pretrained=False, amp="fp32", fused_adamw=False, log_every=1)
            contract = {"synthetic_cpu_fixture": True, "manifest_sha256": "unit-fixture",
                        "size": [32, 64], "depth_scale": 256, "train": 2, "val": 2}
            write_json(Path(cfg["work"]) / "data_contract.json", contract)
            init = root / "parent.pth"
            from model import AnchorFlowEdge as Factory
            parent_model = Factory(pretrained=False,model_name="v5_piecewise")
            torch.save({"model": parent_model.state_dict(), "epoch": 14, "best_rmse": 1.26,
                        "protocol": {"source_sha256": "fixture", "data": contract,
                        "config": {"architecture": "AnchorFlow-v5-PiecewiseSurface", "flow_steps": 3, "encoder": cfg["encoder"]}}}, init)
            from data import digest
            cfg.update(init_checkpoint=str(init), init_sha256=digest(init), init_architecture="AnchorFlow-v5-PiecewiseSurface", parent_epoch=14,parent_cumulative_epoch=44)
            with patch.object(run, "device_setup", return_value=torch.device("cpu")), \
                 patch.object(run, "data_loader", return_value=loader), \
                 patch.object(torch.cuda, "get_device_name", return_value="CPU fixture"):
                run.train(cfg, "metric_kd")
                last = root / "drive" / cfg["run_name"] / "metric_kd" / "last.pth"
                saved = torch.load(last, weights_only=False)
                self.assertEqual(saved["epoch"], 0)
                self.assertEqual(saved["global_step"], 1)
                self.assertTrue(np.isfinite(saved["best_rmse"]))
                # Completed run must resume without adding a duplicate epoch row.
                run.train(cfg, "metric_kd")
                csv = last.with_name("train_log.csv").read_text().splitlines()
                self.assertEqual(len(csv), 2)
                restored, _ = run.load_trained(cfg, "metric_kd", torch.device("cpu"))
                self.assertFalse(restored.training)


if __name__ == "__main__":
    unittest.main()
