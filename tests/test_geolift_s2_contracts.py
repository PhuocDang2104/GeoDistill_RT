import io
import tarfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch.nn.functional as F

from scripts.extract_geolift_teachers import _index, canonical_sample_id
from src.dataset import KITTIDepthCompletionDataset
from src.losses import boundary_ordinal_loss, depth_bin_weight, geolift_loss, scheduled_linear_value
from src.model_geolift_s2 import (
    GeoLiftStudentS2,
    affine_inverse_depth_transport,
    compact_sparse_prior,
    phase_base_offsets,
    phase_pack,
    phase_unpack,
)


class PhasePackingContractTest(unittest.TestCase):
    def test_phase_order_is_q00_q10_q01_q11(self) -> None:
        x = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
        packed = phase_pack(x)
        self.assertEqual(packed.flatten().tolist(), [0.0, 1.0, 2.0, 3.0])
        self.assertTrue(torch.equal(phase_unpack(packed), x))

    def test_align_corners_false_child_offsets(self) -> None:
        expected = torch.tensor(((-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)))
        self.assertTrue(torch.equal(phase_base_offsets(torch.device("cpu"), torch.float32), expected))


class AffineTransportContractTest(unittest.TestCase):
    def test_matches_inverse_depth_plane_normal_form(self) -> None:
        # Plane inverse depth is xi(x,y)=a*x+b*y+c.
        a, b, c = 0.12, -0.07, 0.2
        xs, ys = torch.tensor(0.3), torch.tensor(-0.2)
        xt, yt = torch.tensor(-0.1), torch.tensor(0.4)
        xi_source = a * xs + b * ys + c
        transported = affine_inverse_depth_transport(xi_source, torch.tensor(a), torch.tensor(b), xt, yt, xs, ys)
        expected = a * xt + b * yt + c
        self.assertTrue(torch.allclose(transported, expected, atol=1e-7))


class GeometryTeacherContractTest(unittest.TestCase):
    def test_geometry_archive_id_is_canonicalized(self) -> None:
        raw = "2011_09_29_drive_0071_sync_image_03_0000000915"
        expected = "2011_09_29_drive_0071_sync_image_0000000915_image_03"
        self.assertEqual(canonical_sample_id(raw, "geometry"), expected)
        self.assertEqual(canonical_sample_id(raw, "da"), expected)

    def test_tar_index_normalizes_historical_da_and_geometry_names(self) -> None:
        raw = "2011_09_29_drive_0071_sync_image_03_0000000915"
        expected = "2011_09_29_drive_0071_sync_image_0000000915_image_03"
        with TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "teacher.tar"
            payload = io.BytesIO()
            np.savez(payload, R_G=np.zeros((2, 3), np.float32), C_G=np.ones((2, 3), np.float32))
            data = payload.getvalue()
            with tarfile.open(archive_path, "w") as archive:
                info = tarfile.TarInfo(f"geometry_fused/train/{raw}.npz")
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            self.assertEqual(_index(archive_path, "geometry"), {expected: f"geometry_fused/train/{raw}.npz"})
            self.assertEqual(_index(archive_path, "da"), {expected: f"geometry_fused/train/{raw}.npz"})

    def test_fused_geometry_is_loaded_without_second_normalization(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_id = "sample"
            target = root / "geometry_fused" / "train"
            target.mkdir(parents=True)
            r_g = np.linspace(-1.25, 1.5, 48, dtype=np.float32).reshape(6, 8)
            c_g = np.linspace(0.3, 1.0, 48, dtype=np.float32).reshape(6, 8)
            np.savez(target / f"{sample_id}.npz", R_G=r_g, C_G=c_g)
            dataset = KITTIDepthCompletionDataset.__new__(KITTIDepthCompletionDataset)
            dataset.teacher_root = root
            dataset.split_name = "train"
            dataset.geometry_fallback = False
            dataset.min_depth = 1e-3
            dataset.max_depth = 120.0
            dataset._warned_dmd_geometry_fallback = False

            loaded_r, loaded_c = dataset._load_geometry_teacher(sample_id, r_g.shape)

            np.testing.assert_allclose(loaded_r, r_g)
            np.testing.assert_allclose(loaded_c, c_g)


class CompactSparsePriorContractTest(unittest.TestCase):
    def test_empty_support_stays_invalid_and_finite(self) -> None:
        sparse = torch.zeros(1, 1, 64, 64)
        mask = torch.zeros_like(sparse)
        sparse[..., 8, 8] = 12.0
        mask[..., 8, 8] = 1.0
        _, _, depth, valid, density = compact_sparse_prior(sparse, mask, scale=4, radius=1)
        self.assertTrue(torch.isfinite(depth).all())
        self.assertTrue(torch.equal(depth[valid == 0], torch.zeros_like(depth[valid == 0])))
        self.assertTrue(((density >= 0.0) & (density <= 1.0)).all())


class BalancedAblationLossContractTest(unittest.TestCase):
    def test_sparse_weight_decays_only_after_warmup(self) -> None:
        self.assertAlmostEqual(scheduled_linear_value(0.2, 0.05, 2, 3, 5), 0.2)
        self.assertAlmostEqual(scheduled_linear_value(0.2, 0.05, 3, 3, 5), 0.17)
        self.assertAlmostEqual(scheduled_linear_value(0.2, 0.05, 7, 3, 5), 0.05)

    def test_metric_kd_depth_bin_weights(self) -> None:
        depth = torch.tensor([[[[10.0, 30.0, 50.0, 70.0, 100.0]]]])
        actual = depth_bin_weight(depth, [0, 20, 40, 60, 80, 120], [1.0, 1.0, 1.25, 1.5, 2.0])
        expected = torch.tensor([[[[1.0, 1.0, 1.25, 1.5, 2.0]]]])
        self.assertTrue(torch.equal(actual, expected))

    def test_temperature_ordinal_rewards_correct_far_pair_order(self) -> None:
        x = torch.arange(8, dtype=torch.float32).view(1, 1, 1, 8).expand(1, 1, 4, 8)
        relative = 0.2 * x
        confidence = torch.ones_like(relative)
        rgb = torch.zeros(1, 3, 4, 8)
        good_depth = torch.exp(2.0 + 0.2 * x)
        bad_depth = torch.exp(2.0 - 0.2 * x)
        kwargs = {
            "conf_threshold": 0.4,
            "geom_tau": 0.05,
            "temperature": 0.1,
            "offsets": [1, 2, 4],
            "require_geometry_edge": True,
            "return_stats": True,
        }
        good_loss, good_accuracy, pair_ratio = boundary_ordinal_loss(
            good_depth, relative, confidence, rgb, **kwargs
        )
        bad_loss, bad_accuracy, _ = boundary_ordinal_loss(
            bad_depth, relative, confidence, rgb, **kwargs
        )
        self.assertLess(float(good_loss), float(bad_loss))
        self.assertEqual(float(good_accuracy), 1.0)
        self.assertEqual(float(bad_accuracy), 0.0)
        self.assertGreater(float(pair_ratio), 0.0)

    def test_combined_ablation_weights_enter_geolift_total(self) -> None:
        height, width = 32, 64
        base = torch.linspace(8.0, 80.0, height * width).view(1, 1, height, width)
        predictions = {
            name: F.interpolate(base, scale_factor=1 / scale, mode="area").clone().requires_grad_()
            for name, scale in (("D16", 16), ("D8", 8), ("D4", 4), ("D2", 2), ("D1", 1))
        }
        predictions["D_pre_anchor"] = (base + 0.5).clone().requires_grad_()
        predictions["C1"] = torch.full_like(base, 0.8, requires_grad=True)
        gt_mask = torch.zeros_like(base)
        gt_mask[..., ::4, ::4] = 1.0
        sparse_mask = torch.zeros_like(base)
        sparse_mask[..., ::8, ::8] = 1.0
        relative = torch.log(base)
        batch = {
            "rgb": torch.zeros(1, 3, height, width),
            "gt": base,
            "gt_mask": gt_mask,
            "sparse": base * sparse_mask,
            "mask": sparse_mask,
            "D_cm": base + 0.25,
            "C_cm": torch.ones_like(base),
            "R_G": relative,
            "C_G": torch.ones_like(base),
        }
        loss_cfg = {
            "lambda_gt": 1.0,
            "lambda_S": 0.2,
            "lambda_S_final": 0.05,
            "lambda_range": 0.005,
            "lambda_cm": 0.4,
            "lambda_boundary": 1.0,
            "lambda_cycle": 0.02,
            "lambda_G": 0.03,
            "lambda_ord_in": 1.0,
            "lambda_C": 0.0,
            "multiscale_weights": {"D16": 0.2, "D8": 0.35, "D4": 0.5, "D2": 0.7, "D1": 1.0},
            "range_bins": [0, 20, 40, 60, 80, 120],
            "range_weights": [1, 1.2, 1.5, 2, 2.5],
            "metric_kd_range_weights": [1, 1, 1.25, 1.5, 2],
            "geometry_conf_threshold": 0.4,
            "geometry_min_valid_pixels": 1,
            "min_dense_metric_pixels": 1,
            "ordinal_temperature": 0.05,
            "ordinal_offsets": [1, 4, 8],
            "ordinal_geom_tau": 0.01,
            "ordinal_require_geometry_edge": True,
        }
        schedule_cfg = {
            "add_teacher_epoch": 3,
            "add_geometry_epoch": 5,
            "add_confidence_epoch": 8,
            "add_range_epoch": 3,
            "sparse_decay_start_epoch": 3,
            "teacher_ramp_epochs": 3,
            "geometry_ramp_epochs": 3,
            "confidence_ramp_epochs": 2,
            "range_ramp_epochs": 5,
            "sparse_decay_epochs": 5,
        }
        total, items = geolift_loss(predictions, batch, loss_cfg, schedule_cfg, epoch=5)
        self.assertTrue(torch.isfinite(total))
        self.assertAlmostEqual(items["w_sparse"], 0.11)
        self.assertAlmostEqual(items["w_range"], 0.003)
        self.assertGreater(items["ord_pair_ratio"], 0.0)
        total.backward()
        self.assertTrue(torch.isfinite(predictions["D1"].grad).all())


class GeoLiftInitializationContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(7)
        cls.model = GeoLiftStudentS2().eval()
        b, h, w = 1, 64, 128
        cls.rgb = torch.rand(b, 3, h, w)
        cls.sparse = torch.zeros(b, 1, h, w)
        cls.sparse[:, :, ::8, ::8] = 15.0
        cls.mask = (cls.sparse > 0.0).float()
        fx = fy = 100.0
        cx, cy = w / 2.0, h / 2.0
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        cls.ray = torch.stack(((xx - cx) / fx, (yy - cy) / fy, torch.ones_like(xx)), dim=0).float()[None]
        cls.uv = torch.zeros(b, 2, h, w)
        cls.K = torch.tensor([[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]])
        with torch.inference_mode():
            cls.output = cls.model(cls.rgb, cls.sparse, cls.mask, cls.ray, cls.uv, cls.K)

    def test_tensor_pyramid_and_finite_outputs(self) -> None:
        expected = {
            "D16": (1, 1, 4, 8),
            "D8": (1, 1, 8, 16),
            "D4": (1, 1, 16, 32),
            "D2": (1, 1, 32, 64),
            "D1": (1, 1, 64, 128),
        }
        for key, shape in expected.items():
            self.assertEqual(tuple(self.output[key].shape), shape)
            self.assertTrue(torch.isfinite(self.output[key]).all())

    def test_initial_raylift_is_close_to_inverse_depth_bilinear(self) -> None:
        for parent, child in (("D16", "D8"), ("D8", "D4"), ("D4", "D2"), ("D2", "D1")):
            reference = F.interpolate(self.output[parent].reciprocal(), scale_factor=2, mode="bilinear", align_corners=False).reciprocal()
            self.assertLess(float((self.output[child] - reference).abs().max()), 2e-3)

    def test_hard_sparse_anchor_is_exact(self) -> None:
        error = ((self.output["D_full"] - self.sparse) * self.mask).abs().max()
        self.assertEqual(float(error), 0.0)
        self.assertEqual(float(((self.output["C_full"] - 1.0) * self.mask).abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
