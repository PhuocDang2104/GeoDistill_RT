import unittest

import torch

from src.losses import geolift_s3_loss
from src.metrics import DepthPredictionAudit
from src.model_geolift_decoder_v2 import GeoLiftDecoderV2
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


class GeoLiftDecoderV2ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(23)
        cls.base = GeoLiftStudentS3Lite(encoder_pretrained=False).eval()
        cls.model = GeoLiftDecoderV2(encoder_pretrained=False).eval()
        cls.load_result = cls.model.load_state_dict(cls.base.state_dict(), strict=False)
        b, h, w = 1, 64, 128
        cls.rgb = torch.rand(b, 3, h, w)
        cls.sparse = torch.zeros(b, 1, h, w)
        cls.sparse[:, :, ::8, ::8] = 18.0
        cls.mask = (cls.sparse > 0.0).float()
        cls.ray = torch.zeros(b, 3, h, w)
        cls.uv = torch.zeros(b, 2, h, w)
        cls.K = torch.tensor([[[100.0, 0.0, w / 2], [0.0, 100.0, h / 2], [0.0, 0.0, 1.0]]])
        with torch.inference_mode():
            cls.base_output = cls.base(cls.rgb, cls.sparse, cls.mask, cls.ray, cls.uv, cls.K)
            cls.output = cls.model(cls.rgb, cls.sparse, cls.mask, cls.ray, cls.uv, cls.K)

    def test_s3_checkpoint_has_only_expected_new_missing_keys(self) -> None:
        allowed = ("lift8_4.metric_", "lift8_4.sparse_metric_project", "f2_project", "f2_alpha")
        self.assertFalse(self.load_result.unexpected_keys)
        self.assertTrue(self.load_result.missing_keys)
        self.assertTrue(all(any(key.startswith(prefix) for prefix in allowed) for key in self.load_result.missing_keys))

    def test_warm_start_is_an_exact_noop(self) -> None:
        for stage in ("D16", "D8", "D4", "D2", "D1", "D_full"):
            self.assertEqual(float((self.output[stage] - self.base_output[stage]).abs().max()), 0.0)
        self.assertAlmostEqual(float(self.output["metric_gate_4"].mean()), 0.05, places=6)
        self.assertEqual(float(self.output["metric_delta_4"].abs().max()), 0.0)
        self.assertEqual(float(self.output["f2_alpha"]), 0.0)

    def test_new_parameter_budget_is_below_two_percent(self) -> None:
        base_parameters = sum(parameter.numel() for parameter in self.base.parameters())
        new_parameters = sum(parameter.numel() for parameter in self.model.parameters())
        self.assertLess((new_parameters - base_parameters) / base_parameters, 0.02)

    def test_new_heads_receive_gradient_from_unchanged_s3_loss(self) -> None:
        self.model.train()
        output = self.model(self.rgb, self.sparse, self.mask, self.ray, self.uv, self.K)
        target = torch.full_like(self.sparse, 20.0)
        loss, _ = geolift_s3_loss(
            output,
            {"gt": target, "gt_mask": torch.ones_like(target), "sparse": self.sparse, "mask": self.mask},
            {},
        )
        self.model.zero_grad(set_to_none=True)
        loss.backward()
        self.assertGreater(float(self.model.lift8_4.metric_delta.weight.grad.abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(self.model.f2_alpha.grad))
        self.model.eval()


class DepthPredictionAuditContractTest(unittest.TestCase):
    def test_localizes_near_zero_stage_and_records_outlier_context(self) -> None:
        height, width = 8, 16
        gt = torch.full((1, 1, height, width), 10.0)
        predictions = {
            "D8": torch.full((1, 1, 1, 2), 10.0),
            "D4": torch.full((1, 1, 2, 4), 10.0),
            "D2": torch.full((1, 1, 4, 8), 10.0),
            "D1": torch.full_like(gt, 10.0),
            "D_full": torch.full_like(gt, 10.0),
        }
        predictions["D1"][0, 0, 2, 3] = 0.001
        audit = DepthPredictionAudit(top_k=3)
        audit.update(predictions, {"gt": gt, "gt_mask": torch.ones_like(gt), "sample_id": ["sample"]})
        report = audit.report()
        self.assertEqual(report["stages"]["D8"]["low_depth_counts"]["lt_0.01m"], 0)
        self.assertEqual(report["stages"]["D1"]["low_depth_counts"]["lt_0.01m"], 1)
        self.assertGreater(report["inverse_depth"]["D1"]["irmse_raw_1_per_km"], 1000.0)
        outlier = report["top_inverse_depth_outliers"][0]
        self.assertEqual((outlier["sample_id"], outlier["u"], outlier["v"]), ("sample", 3, 2))
        self.assertEqual(outlier["stage_prediction_m"]["D4"], 10.0)


if __name__ == "__main__":
    unittest.main()
