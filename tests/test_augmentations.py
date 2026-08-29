from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.augmentations import DepthCompletionAugmentor
from src.dataset import KITTIDepthCompletionDataset
from src.utils import make_ray_map, make_uv_map


def sample_fixture(height: int = 8, width: int = 12) -> dict[str, np.ndarray | str | tuple[int, int]]:
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    dense = (1.0 + yy * width + xx).astype(np.float32)
    sparse = np.zeros((height, width), dtype=np.float32)
    sparse[1::2, 1::3] = dense[1::2, 1::3]
    K = np.asarray([[10.0, 0.0, 3.25], [0.0, 11.0, 2.75], [0.0, 0.0, 1.0]], dtype=np.float32)
    return {
        "sample_id": "fixture",
        "rgb": np.stack((dense, dense + 1.0, dense + 2.0), axis=-1).astype(np.uint8),
        "sparse": sparse,
        "mask": (sparse > 0.0).astype(np.float32),
        "gt": dense.copy(),
        "gt_mask": np.ones_like(dense, dtype=np.float32),
        "D_cm": dense.copy(),
        "C_cm": np.ones_like(dense, dtype=np.float32),
        "R_G": dense.copy(),
        "C_G": np.ones_like(dense, dtype=np.float32),
        "K": K,
        "ray": make_ray_map(K, height, width),
        "uv": make_uv_map(height, width),
        "orig_hw": (height, width),
    }


class AugmentationContractTests(unittest.TestCase):
    def test_a1_horizontal_flip_keeps_teacher_alignment_and_updates_camera(self) -> None:
        sample = sample_fixture()
        original_dense = np.asarray(sample["D_cm"]).copy()
        original_sparse = np.asarray(sample["sparse"]).copy()
        augmentor = DepthCompletionAugmentor(
            {
                "enabled": True,
                "stage": "A1",
                "horizontal_flip_prob": 1.0,
            }
        )
        output = augmentor(sample, rng=np.random.default_rng(7))
        width = np.asarray(output["rgb"]).shape[1]

        np.testing.assert_array_equal(output["D_cm"], original_dense[:, ::-1])
        np.testing.assert_array_equal(output["R_G"], original_dense[:, ::-1])
        np.testing.assert_array_equal(output["sparse"], original_sparse[:, ::-1])
        self.assertAlmostEqual(float(output["K"][0, 2]), width - 1 - 3.25, places=6)
        np.testing.assert_allclose(output["ray"], make_ray_map(output["K"], 8, 12), atol=1e-7)
        np.testing.assert_array_equal(output["uv"], make_uv_map(8, 12))
        self.assertEqual(float(output["augmentation_params"][4]), 1.0)

    def test_a2_sparse_dropout_changes_only_student_sparse_input(self) -> None:
        sample = sample_fixture()
        teacher_before = np.asarray(sample["D_cm"]).copy()
        gt_before = np.asarray(sample["gt"]).copy()
        points_before = int(np.count_nonzero(sample["sparse"]))
        augmentor = DepthCompletionAugmentor(
            {
                "enabled": True,
                "stage": "A2",
                "horizontal_flip_prob": 0.0,
                "sparse_dropout": {
                    "enabled": True,
                    "min_rate": 0.5,
                    "max_rate": 0.5,
                    "min_points": 2,
                },
            }
        )
        output = augmentor(sample, rng=np.random.default_rng(11))
        points_after = int(np.count_nonzero(output["sparse"]))

        self.assertLess(points_after, points_before)
        self.assertGreaterEqual(points_after, 2)
        np.testing.assert_array_equal(output["mask"], np.asarray(output["sparse"]) > 0.0)
        np.testing.assert_array_equal(output["D_cm"], teacher_before)
        np.testing.assert_array_equal(output["gt"], gt_before)
        self.assertEqual(int(output["augmentation_params"][6]), points_before)
        self.assertEqual(int(output["augmentation_params"][7]), points_after)

    def test_a3_scale_crop_keeps_dense_teachers_aligned_and_sparse_points_unique(self) -> None:
        sample = sample_fixture(height=10, width=14)
        points_before = int(np.count_nonzero(sample["sparse"]))
        augmentor = DepthCompletionAugmentor(
            {
                "enabled": True,
                "stage": "A3",
                "horizontal_flip_prob": 0.0,
                "scale_jitter": {"enabled": True, "min_scale": 1.1, "max_scale": 1.1},
            }
        )
        output = augmentor(sample, rng=np.random.default_rng(19))
        params = output["augmentation_params"]

        self.assertEqual(np.asarray(output["rgb"]).shape[:2], (10, 14))
        self.assertEqual(np.asarray(output["D_cm"]).shape, (10, 14))
        gt_valid = np.asarray(output["gt_mask"]) > 0.5
        np.testing.assert_allclose(np.asarray(output["D_cm"])[gt_valid], np.asarray(output["gt"])[gt_valid], atol=1e-6)
        np.testing.assert_allclose(np.asarray(output["C_cm"])[gt_valid], 1.0, atol=1e-6)
        self.assertTrue(np.isfinite(output["R_G"]).all())
        self.assertTrue(np.isfinite(output["C_G"]).all())
        self.assertLessEqual(int(np.count_nonzero(output["sparse"])), points_before)
        np.testing.assert_array_equal(output["mask"], np.asarray(output["sparse"]) > 0.0)

        scale_x, scale_y, crop_x, crop_y = map(float, params[:4])
        self.assertAlmostEqual(float(output["K"][0, 0]), 10.0 * scale_x, places=5)
        self.assertAlmostEqual(float(output["K"][1, 1]), 11.0 * scale_y, places=5)
        self.assertAlmostEqual(float(output["K"][0, 2]), (3.25 + 0.5) * scale_x - 0.5 - crop_x, places=5)
        self.assertAlmostEqual(float(output["K"][1, 2]), (2.75 + 0.5) * scale_y - 0.5 - crop_y, places=5)
        np.testing.assert_allclose(output["ray"], make_ray_map(output["K"], 10, 14), atol=1e-7)

    def test_disabled_augmentation_returns_the_original_sample(self) -> None:
        sample = sample_fixture()
        output = DepthCompletionAugmentor({"enabled": False})(sample, rng=np.random.default_rng(0))
        self.assertIs(output, sample)
        self.assertNotIn("augmentation_params", output)

    def test_zoom_out_is_rejected_until_padding_contract_is_implemented(self) -> None:
        with self.assertRaises(ValueError):
            DepthCompletionAugmentor(
                {
                    "enabled": True,
                    "scale_jitter": {"enabled": True, "min_scale": 0.9, "max_scale": 1.1},
                }
            )

    def test_dataset_loads_teachers_before_shared_augmentation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            split_root = root / "splits"
            teacher_root = root / "teachers"
            for path in (
                data_root / "rgb",
                data_root / "sparse",
                data_root / "gt",
                split_root,
                teacher_root / "metric_coarse" / "train",
                teacher_root / "geometry_fused" / "train",
            ):
                path.mkdir(parents=True, exist_ok=True)

            height, width = 8, 12
            dense = np.arange(1, height * width + 1, dtype=np.float32).reshape(height, width)
            rgb = np.stack((dense, dense, dense), axis=-1).astype(np.uint8)
            sparse = np.zeros_like(dense)
            sparse[::2, ::3] = dense[::2, ::3]
            cv2.imwrite(str(data_root / "rgb" / "sample.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(data_root / "sparse" / "sample.png"), np.rint(sparse * 256.0).astype(np.uint16))
            cv2.imwrite(str(data_root / "gt" / "sample.png"), np.rint(dense * 256.0).astype(np.uint16))
            np.savez_compressed(
                teacher_root / "metric_coarse" / "train" / "sample.npz", D_cm=dense, C_cm=np.ones_like(dense)
            )
            np.savez_compressed(
                teacher_root / "geometry_fused" / "train" / "sample.npz", R_G=dense, C_G=np.ones_like(dense)
            )
            record = {
                "id": "sample",
                "rgb": "rgb/sample.png",
                "sparse": "sparse/sample.png",
                "gt": "gt/sample.png",
                "K": [10.0, 11.0, 3.25, 2.75],
            }
            (split_root / "train.txt").write_text(json.dumps(record) + "\n", encoding="utf-8")

            dataset = KITTIDepthCompletionDataset(
                data_root=data_root,
                split_root=split_root,
                split_file="train.txt",
                split_name="train",
                image_size=(height, width),
                teacher_root=teacher_root,
                load_teacher=True,
                load_geometry=True,
                geometry_fallback=False,
                calibrate_metric_teacher=False,
                augmentation={"enabled": True, "stage": "A1", "horizontal_flip_prob": 1.0},
            )
            output = dataset[0]
            self.assertIn("augmentation_params", output)
            self.assertEqual(tuple(output["D_teacher"].shape), (1, height // 4, width // 4))
            np.testing.assert_allclose(output["D_cm"][0].numpy(), output["gt"][0].numpy(), atol=1e-6)
            np.testing.assert_allclose(output["R_G"][0].numpy(), dense[:, ::-1], atol=1e-6)
            self.assertTrue(bool((output["mask"] == (output["sparse"] > 0.0)).all()))


if __name__ == "__main__":
    unittest.main()
