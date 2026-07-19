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
