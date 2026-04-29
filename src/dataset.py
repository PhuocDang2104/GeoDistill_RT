from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import (
    DEFAULT_KITTI_K,
    load_intrinsics_from_calib,
    make_ray_map,
    make_uv_map,
    read_depth,
    read_image_rgb,
    read_split,
    resize_depth,
    resize_rgb,
    safe_sample_id,
    scale_intrinsics,
)


@dataclass(frozen=True)
class SampleInfo:
    sample_id: str
    rgb_path: Path
    sparse_path: Path | None
    gt_path: Path | None
    K: np.ndarray | None = None
    calib_path: Path | None = None


class KITTIDepthCompletionDataset(Dataset):
    """KITTI Depth Completion dataset adapter.

    Tensor output shapes:
      rgb: [3,H,W], sparse: [1,H,W], mask: [1,H,W]
      ray: [3,H,W], uv: [2,H,W], K: [3,3]
      D_teacher/C_teacher when loaded: [1,H/4,W/4] by default.
    """

    def __init__(
        self,
        data_root: str | Path,
        split_root: str | Path,
        split_file: str,
        split_name: str,
        image_size: tuple[int, int] | list[int] | None = None,
        output_scale: int = 4,
        depth_scale: float = 256.0,
        teacher_root: str | Path | None = None,
        load_teacher: bool = False,
        return_tensors: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.split_root = Path(split_root)
        self.split_file = split_file
        self.split_name = split_name
        self.image_size = tuple(image_size) if image_size is not None else None
        self.output_scale = int(output_scale)
        self.depth_scale = float(depth_scale)
        self.teacher_root = Path(teacher_root) if teacher_root is not None else None
        self.load_teacher = load_teacher
        self.return_tensors = return_tensors

        lines = read_split(self.split_root, split_file)
        self.samples = [self._parse_line(line) for line in lines]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.load_sample_np(index)
        if not self.return_tensors:
            return sample

        rgb = torch.from_numpy(sample["rgb"].transpose(2, 0, 1)).float() / 255.0
        sparse = torch.from_numpy(sample["sparse"][None]).float()
        mask = torch.from_numpy(sample["mask"][None]).float()
        gt = torch.from_numpy(sample["gt"][None]).float()
        gt_mask = torch.from_numpy(sample["gt_mask"][None]).float()
        ray = torch.from_numpy(sample["ray"]).float()
        uv = torch.from_numpy(sample["uv"]).float()
        K = torch.from_numpy(sample["K"]).float()

        out: dict[str, Any] = {
            "sample_id": sample["sample_id"],
            "rgb": rgb,
            "sparse": sparse,
            "mask": mask,
            "gt": gt,
            "gt_mask": gt_mask,
            "ray": ray,
            "uv": uv,
            "K": K,
            "orig_hw": torch.tensor(sample["orig_hw"], dtype=torch.long),
        }
        if self.load_teacher:
            D_teacher, C_teacher = self._load_teacher(sample["sample_id"], sample["rgb"].shape[:2])
            out["D_teacher"] = torch.from_numpy(D_teacher[None]).float()
            out["C_teacher"] = torch.from_numpy(C_teacher[None]).float()
        return out

    def load_sample_np(self, index: int) -> dict[str, Any]:
        info = self.samples[index]
        rgb0 = read_image_rgb(info.rgb_path)
        orig_h, orig_w = rgb0.shape[:2]

        sparse = read_depth(info.sparse_path, self.depth_scale)
        gt = read_depth(info.gt_path, self.depth_scale)
        if sparse is None:
            sparse = np.zeros((orig_h, orig_w), dtype=np.float32)
        if gt is None:
            gt = np.zeros((orig_h, orig_w), dtype=np.float32)

        if sparse.shape != (orig_h, orig_w):
            sparse = resize_depth(sparse, (orig_h, orig_w))
        if gt.shape != (orig_h, orig_w):
            gt = resize_depth(gt, (orig_h, orig_w))

        K = self._sample_intrinsics(info, (orig_h, orig_w))
        target_size = self.image_size
        if target_size is not None:
            rgb = resize_rgb(rgb0, target_size)
            sparse = resize_depth(sparse, target_size)
            gt = resize_depth(gt, target_size)
            K = scale_intrinsics(K, (orig_h, orig_w), target_size)
        else:
            rgb = rgb0

        assert sparse is not None and gt is not None
        h, w = rgb.shape[:2]
        mask = ((sparse > 0.0) & np.isfinite(sparse)).astype(np.float32)
        gt_mask = ((gt > 0.0) & np.isfinite(gt)).astype(np.float32)
        ray = make_ray_map(K, h, w)
        uv = make_uv_map(h, w)

        return {
            "sample_id": info.sample_id,
            "rgb": rgb.astype(np.uint8),
            "sparse": sparse.astype(np.float32),
            "mask": mask,
            "gt": gt.astype(np.float32),
            "gt_mask": gt_mask,
            "K": K.astype(np.float32),
            "ray": ray,
            "uv": uv,
            "orig_hw": (orig_h, orig_w),
            "rgb_path": str(info.rgb_path),
            "sparse_path": str(info.sparse_path) if info.sparse_path is not None else "",
            "gt_path": str(info.gt_path) if info.gt_path is not None else "",
        }

    def _load_teacher(self, sample_id: str, image_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        h, w = image_hw
        th, tw = h // self.output_scale, w // self.output_scale
        if self.teacher_root is None:
            return np.zeros((th, tw), np.float32), np.zeros((th, tw), np.float32)
        path = self.teacher_root / "fused" / self.split_name / f"{sample_id}.npz"
        if not path.exists():
            return np.zeros((th, tw), np.float32), np.zeros((th, tw), np.float32)
        with np.load(path) as data:
            D = data["D_teacher"].astype(np.float32)
            C = data["C_teacher"].astype(np.float32)
        return D, C

    def _sample_intrinsics(self, info: SampleInfo, image_hw: tuple[int, int]) -> np.ndarray:
        if info.K is not None:
            return info.K.astype(np.float32)
        if info.calib_path is not None:
            return load_intrinsics_from_calib(info.calib_path)
        return DEFAULT_KITTI_K.copy()

    def _parse_line(self, line: str) -> SampleInfo:
        if line.startswith("{"):
            record = json.loads(line)
            sample_id = safe_sample_id(record.get("id") or record.get("sample_id") or record["rgb"])
            k_value = record.get("K") or record.get("intrinsics")
            K = self._parse_intrinsics_value(k_value) if k_value is not None else None
            calib = self._resolve_path(record["calib"]) if "calib" in record else None
            return SampleInfo(
                sample_id=sample_id,
                rgb_path=self._resolve_path(record["rgb"]),
                sparse_path=self._resolve_path(record.get("sparse")),
                gt_path=self._resolve_path(record.get("gt") or record.get("depth")),
                K=K,
                calib_path=calib,
            )

        tokens = line.split()
        if len(tokens) >= 8:
            sample_id = safe_sample_id(tokens[0])
            rgb, sparse, gt = tokens[1], tokens[2], tokens[3]
            K = np.array(
                [[float(tokens[4]), 0.0, float(tokens[6])], [0.0, float(tokens[5]), float(tokens[7])], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            return SampleInfo(sample_id, self._resolve_path(rgb), self._resolve_path(sparse), self._resolve_path(gt), K=K)

        if len(tokens) == 7:
            rgb, sparse, gt = tokens[0], tokens[1], tokens[2]
            sample_id = safe_sample_id(rgb)
            K = np.array(
                [[float(tokens[3]), 0.0, float(tokens[5])], [0.0, float(tokens[4]), float(tokens[6])], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            return SampleInfo(sample_id, self._resolve_path(rgb), self._resolve_path(sparse), self._resolve_path(gt), K=K)

        if len(tokens) == 5:
            sample_id, rgb, sparse, gt, calib = tokens
            return SampleInfo(
                safe_sample_id(sample_id),
                self._resolve_path(rgb),
                self._resolve_path(sparse),
                self._resolve_path(gt),
                calib_path=self._resolve_path(calib),
            )

        if len(tokens) == 4:
            if self._looks_like_file(tokens[0]) and self._looks_like_file(tokens[1]):
                rgb, sparse, gt, calib = tokens
                return SampleInfo(
                    safe_sample_id(rgb),
                    self._resolve_path(rgb),
                    self._resolve_path(sparse),
                    self._resolve_path(gt),
                    calib_path=self._resolve_path(calib),
                )
            sample_id, rgb, sparse, gt = tokens
            return SampleInfo(safe_sample_id(sample_id), self._resolve_path(rgb), self._resolve_path(sparse), self._resolve_path(gt))

        if len(tokens) == 3:
            rgb, sparse, gt = tokens
            return SampleInfo(safe_sample_id(rgb), self._resolve_path(rgb), self._resolve_path(sparse), self._resolve_path(gt))

        if len(tokens) == 2:
            rgb, sparse = tokens
            return SampleInfo(safe_sample_id(rgb), self._resolve_path(rgb), self._resolve_path(sparse), None)

        if len(tokens) == 1:
            return self._discover_sample(tokens[0])

        raise ValueError(f"Unsupported split line: {line}")

    @staticmethod
    def _parse_intrinsics_value(value: Any) -> np.ndarray:
        arr = np.array(value, dtype=np.float32)
        if arr.shape == (3, 3):
            return arr
        if arr.size >= 4:
            fx, fy, cx, cy = arr.reshape(-1)[:4]
            return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        raise ValueError(f"Unsupported intrinsics value: {value}")

    def _resolve_path(self, token: str | None) -> Path | None:
        if token is None or token == "" or token.lower() == "none":
            return None
        p = Path(token)
        if p.is_absolute():
            return p
        candidates = [
            self.data_root / p,
            self.data_root / self.split_name / p,
            self.split_root / p,
            Path.cwd() / p,
        ]
        for c in candidates:
            if c.exists():
                return c.resolve()
        return (self.data_root / p).resolve()

    @staticmethod
    def _looks_like_file(token: str) -> bool:
        return Path(token).suffix.lower() in {".png", ".jpg", ".jpeg", ".npy", ".npz", ".txt"}

    def _discover_sample(self, token: str) -> SampleInfo:
        root = self.data_root / self.split_name
        stem = Path(token).stem
        if not root.exists():
            maybe_rgb = self._resolve_path(token)
            if maybe_rgb is None:
                raise FileNotFoundError(f"Cannot discover sample {token}: {root} does not exist")
            return SampleInfo(safe_sample_id(token), maybe_rgb, None, None)

        matches = [p for p in root.rglob(f"*{stem}*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".npy", ".npz"}]
        rgb = self._best_match(matches, ("image", "rgb", "color"), reject=("depth", "sparse", "velodyne", "groundtruth", "gt"))
        sparse = self._best_match(matches, ("sparse", "velodyne", "lidar"), reject=("groundtruth", "gt"))
        gt = self._best_match(matches, ("groundtruth", "gt"), reject=("sparse", "velodyne"))
        if rgb is None:
            raise FileNotFoundError(f"Could not discover RGB for sample {token} under {root}")
        return SampleInfo(safe_sample_id(token), rgb, sparse, gt)

    @staticmethod
    def _best_match(paths: list[Path], prefer: tuple[str, ...], reject: tuple[str, ...]) -> Path | None:
        best: tuple[int, Path] | None = None
        for p in paths:
            s = str(p).lower().replace("\\", "/")
            if any(r in s for r in reject):
                continue
            score = sum(1 for key in prefer if key in s)
            if score <= 0:
                continue
            candidate = (score, p)
            if best is None or candidate[0] > best[0]:
                best = candidate
        return best[1] if best is not None else None
