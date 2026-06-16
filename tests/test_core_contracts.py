import unittest

import numpy as np
import torch

from src.model_geort import GeoRTStudentS
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


if __name__ == "__main__":
    unittest.main()
