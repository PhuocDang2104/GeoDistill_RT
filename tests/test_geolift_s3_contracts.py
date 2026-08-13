import unittest

import torch

from src.losses import geolift_s3_loss
from src.model_geolift_s3 import GeoLiftStudentS3Lite


class GeoLiftS3ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(13)
        cls.model = GeoLiftStudentS3Lite(encoder_pretrained=False).train()
        b, h, w = 1, 64, 128
        cls.rgb = torch.rand(b, 3, h, w)
        cls.sparse = torch.zeros(b, 1, h, w)
        cls.sparse[:, :, ::8, ::8] = 15.0
        cls.mask = (cls.sparse > 0.0).float()
        cls.ray = torch.zeros(b, 3, h, w)
        cls.uv = torch.zeros(b, 2, h, w)
        cls.K = torch.tensor([[[100.0, 0.0, w / 2], [0.0, 100.0, h / 2], [0.0, 0.0, 1.0]]])
        cls.output = cls.model(cls.rgb, cls.sparse, cls.mask, cls.ray, cls.uv, cls.K)

    def test_depth_pyramid_and_hard_anchor(self) -> None:
        expected = {
            "D16": (1, 1, 4, 8),
            "D8": (1, 1, 8, 16),
            "D4": (1, 1, 16, 32),
            "D2": (1, 1, 32, 64),
            "D1": (1, 1, 64, 128),
        }
        for name, shape in expected.items():
            self.assertEqual(tuple(self.output[name].shape), shape)
            self.assertTrue(torch.isfinite(self.output[name]).all())
        anchor_error = ((self.output["D_full"] - self.sparse) * self.mask).abs().max()
        self.assertEqual(float(anchor_error.detach()), 0.0)

    def test_all_residual_gates_start_near_five_percent(self) -> None:
        for name in ("residual_gate_8", "residual_gate_4", "residual_gate_2", "residual_gate_1"):
            mean = float(self.output[name].detach().mean())
            self.assertAlmostEqual(mean, 0.05, places=5)

    def test_simple_loss_needs_no_teacher_and_backpropagates(self) -> None:
        target = torch.full_like(self.sparse, 20.0)
        batch = {
            "gt": target,
            "gt_mask": torch.ones_like(target),
            "sparse": self.sparse,
            "mask": self.mask,
        }
        total, items = geolift_s3_loss(self.output, batch, {})
        self.assertTrue(torch.isfinite(total))
        self.assertEqual(items["w_metric"], 1.0)
        self.assertEqual(items["w_log"], 0.2)
        self.assertEqual(items["w_sparse"], 0.1)
        self.assertEqual(items["w_edge"], 0.05)
        total.backward()
        self.assertTrue(
            all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in self.model.parameters())
        )


if __name__ == "__main__":
    unittest.main()
