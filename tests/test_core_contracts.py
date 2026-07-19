import unittest

import numpy as np
import torch

from src.model_geort import AdaptiveSparseAnchor, EfficientFusion, GeoRTStudentS, LightStem
from src.metrics import depth_metrics_torch
from src.sparse_propagation import local_sparse_propagation
from src.teachers.generate_teachers import _calibrate_depth_to_gt


class TeacherCalibrationTest(unittest.TestCase):
    def test_calibrates_raw_depth_to_gt(self) -> None:
        gt = np.linspace(2.0, 40.0, 64, dtype=np.float32).reshape(8, 8)
        raw = 2.0 * gt + 3.0
        mask = np.ones_like(gt, dtype=np.float32)

        calibrated, gamma, delta, count, applied = _calibrate_depth_to_gt(raw, gt, mask, min_points=4)

        self.assertTrue(applied)
        self.assertEqual(count, 64)
        self.assertAlmostEqual(gamma, 0.5, places=5)
        self.assertAlmostEqual(delta, -1.5, places=5)
        np.testing.assert_allclose(calibrated, gt, rtol=1e-5, atol=1e-5)


class DepthMetricsTest(unittest.TestCase):
    def test_reports_kitti_inverse_depth_units(self) -> None:
        pred = torch.tensor([[[[2.0, 4.0]]]])
        target = torch.tensor([[[[1.0, 2.0]]]])
        metrics = depth_metrics_torch(pred, target)

        expected_imae = (500.0 + 250.0) / 2.0
        expected_irmse = ((500.0**2 + 250.0**2) / 2.0) ** 0.5
        self.assertAlmostEqual(metrics["imae"], expected_imae, places=5)
        self.assertAlmostEqual(metrics["irmse"], expected_irmse, places=4)


class SparsePropagationTest(unittest.TestCase):
    def test_local_sparse_propagation_returns_finite_coarse_map(self) -> None:
        sparse = torch.zeros(1, 1, 16, 16)
        mask = torch.zeros_like(sparse)
        sparse[:, :, 8, 8] = 12.0
        mask[:, :, 8, 8] = 1.0

        out = local_sparse_propagation(sparse, mask, scale=4)

        self.assertEqual(tuple(out.shape), (1, 1, 4, 4))
        self.assertTrue(torch.isfinite(out).all().item())
        self.assertGreater(float(out.min()), 0.0)


class EncoderContractTest(unittest.TestCase):
    def test_invalid_timm_encoder_raises(self) -> None:
        with self.assertRaises(Exception):
            GeoRTStudentS(
                encoder="definitely_missing_mobilevit_model",
                fusion_channels=16,
                e4_channels=16,
                e8_channels=24,
                e16_channels=32,
            )


class EfficientFusionContractTest(unittest.TestCase):
    def test_reduces_and_preserves_spatial_shape(self) -> None:
        fusion = EfficientFusion(in_ch=52, out_ch=32).eval()
        x = torch.randn(2, 52, 8, 12)

        with torch.no_grad():
            out = fusion(x)

        self.assertEqual(tuple(out.shape), (2, 32, 8, 12))
        self.assertEqual(fusion[1][0].groups, 32)


class LightStemContractTest(unittest.TestCase):
    def test_preserves_shape_and_uses_depthwise_spatial_mix(self) -> None:
        stem = LightStem(3, 24).eval()
        x = torch.randn(2, 3, 8, 12)

        with torch.no_grad():
            out = stem(x)

        self.assertEqual(tuple(out.shape), (2, 24, 8, 12))
        self.assertEqual(stem[1][0][0].groups, 24)


class AdaptiveSparseAnchorContractTest(unittest.TestCase):
    def test_starts_from_fixed_lambda_and_only_changes_valid_points(self) -> None:
        anchor = AdaptiveSparseAnchor(lambda_min=0.5, lambda_init=0.7).eval()
        depth = torch.full((1, 1, 4, 5), 10.0)
        confidence = torch.full_like(depth, 0.4)
        sparse = torch.zeros_like(depth)
        mask = torch.zeros_like(depth)
        sparse[..., 2, 3] = 20.0
        mask[..., 2, 3] = 1.0
        rgb = torch.rand(1, 3, 4, 5)

        with torch.no_grad():
            out, conf, anchor_lambda = anchor(depth, confidence, sparse, mask, rgb)

        self.assertTrue(torch.allclose(out[mask == 0], depth[mask == 0]))
        self.assertAlmostEqual(float(anchor_lambda[..., 2, 3]), 0.7, places=5)
        self.assertAlmostEqual(float(out[..., 2, 3]), 17.0, places=5)
        self.assertAlmostEqual(float(conf[..., 2, 3]), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
