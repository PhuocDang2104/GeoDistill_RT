from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
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


LOGGER = logging.getLogger("geort")


@dataclass(frozen=True)
class SampleInfo:
    sample_id: str
    rgb_path: Path
    sparse_path: Path | None
    gt_path: Path | None
    K: np.ndarray | None = None
    calib_path: Path | None = None


def _as_hw(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            arr = arr[0]
    return arr.astype(np.float32)


def _resize_hw(array: np.ndarray, shape_hw: tuple[int, int], interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    arr = _as_hw(array)
    if arr.shape == shape_hw:
        return arr.astype(np.float32)
    h, w = shape_hw
    return cv2.resize(arr, (w, h), interpolation=interpolation).astype(np.float32)


def _load_first_npz_key(
    path: Path,
    keys: tuple[str, ...],
    shape_hw: tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        with np.load(path) as data:
            for key in keys:
                if key in data:
                    return _resize_hw(data[key], shape_hw, interpolation)
    except Exception:
        return None
    return None


def _valid_depth(depth: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    return np.isfinite(depth) & (depth > min_depth) & (depth < max_depth)


def _calibrate_to_gt(
    depth: np.ndarray,
    gt: np.ndarray,
    gt_valid: np.ndarray,
    min_depth: float,
    max_depth: float,
    min_points: int = 128,
) -> np.ndarray:
    valid = gt_valid & _valid_depth(depth, min_depth, max_depth)
    if int(valid.sum()) < int(min_points):
        return depth.astype(np.float32)

    x = depth[valid].astype(np.float64)
    y = gt[valid].astype(np.float64)
    residual = np.abs(x - y)
    if residual.size >= 512:
        keep = residual <= np.percentile(residual, 90.0)
        x = x[keep]
        y = y[keep]
    if x.size < int(min_points):
        return depth.astype(np.float32)

    A = np.stack([x, np.ones_like(x)], axis=1)
    try:
        gamma, delta = np.linalg.lstsq(A, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return depth.astype(np.float32)
    if not np.isfinite(gamma) or not np.isfinite(delta) or gamma <= 0.05 or gamma > 20.0:
        return depth.astype(np.float32)
    calibrated = gamma * depth.astype(np.float32) + np.float32(delta)
    calibrated[~np.isfinite(calibrated)] = 0.0
    return calibrated.astype(np.float32)


def _nearest_sparse_confidence(
    depth: np.ndarray,
    sparse: np.ndarray,
    sparse_mask: np.ndarray,
    min_depth: float,
    max_depth: float,
    decay: float,
    blend_radius: float,
) -> np.ndarray:
    anchor = (sparse_mask > 0.5) & _valid_depth(sparse, min_depth, max_depth) & _valid_depth(depth, min_depth, max_depth)
    if int(anchor.sum()) < 1:
        return np.ones_like(depth, dtype=np.float32)

    rel_error = np.abs(depth[anchor] - sparse[anchor]) / np.clip(sparse[anchor], min_depth, max_depth)
    anchor_conf = np.exp(-float(decay) * np.clip(rel_error, 0.0, 10.0)).astype(np.float32)
    try:
        src = np.ones(depth.shape, dtype=np.uint8)
        src[anchor] = 0
        dist, labels = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        max_label = int(labels.max())
        label_conf = np.ones(max_label + 1, dtype=np.float32)
        anchor_labels = labels[anchor].reshape(-1)
        for lab, conf in zip(anchor_labels, anchor_conf.reshape(-1)):
            if 0 <= int(lab) <= max_label:
                label_conf[int(lab)] = min(label_conf[int(lab)], float(conf))
        dense_conf = label_conf[labels]
        if blend_radius > 0:
            far_blend = 1.0 - np.exp(-dist.astype(np.float32) / float(blend_radius))
            dense_conf = dense_conf + (1.0 - dense_conf) * far_blend
        return np.clip(dense_conf, 0.0, 1.0).astype(np.float32)
    except Exception:
        out = np.ones_like(depth, dtype=np.float32)
        out[anchor] = anchor_conf
        return out


def _metric_confidence(
    depth: np.ndarray,
    sparse: np.ndarray,
    sparse_mask: np.ndarray,
    min_depth: float,
    max_depth: float,
    c_min: float,
    sparse_decay: float,
    range_decay: float,
    blend_radius: float,
) -> np.ndarray:
    valid = _valid_depth(depth, min_depth, max_depth)
    range_conf = np.exp(-float(range_decay) * np.clip(depth, 0.0, max_depth) / float(max_depth)).astype(np.float32)
    sparse_conf = _nearest_sparse_confidence(depth, sparse, sparse_mask, min_depth, max_depth, sparse_decay, blend_radius)
    conf = range_conf * sparse_conf
    conf = np.clip(conf, float(c_min), 1.0).astype(np.float32)
    conf[~valid] = 0.0
    return conf


def _robust_normalize(x: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    x = x.astype(np.float32)
    valid_mask = np.isfinite(x) if valid is None else (valid & np.isfinite(x))
    values = x[valid_mask]
    if values.size < 32:
        return np.zeros_like(x, dtype=np.float32)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    std = float(np.std(values))
    scale = max(1e-6, 1.4826 * mad, 0.05 * std)
    out = (x - median) / scale
    out[~np.isfinite(out)] = 0.0
    return np.clip(out, -10.0, 10.0).astype(np.float32)


def _downsample_map(array: np.ndarray, scale: int, interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    h, w = array.shape[-2:]
    out_h = max(1, h // int(scale))
    out_w = max(1, w // int(scale))
    return cv2.resize(_as_hw(array), (out_w, out_h), interpolation=interpolation).astype(np.float32)


class KITTIDepthCompletionDataset(Dataset):
    """KITTI Depth Completion dataset adapter.

    Tensor output shapes:
      rgb: [3,H,W], sparse: [1,H,W], mask: [1,H,W]
      ray: [3,H,W], uv: [2,H,W], K: [3,3]
      D_cm/C_cm when loaded: [1,H,W] metric teacher with GT priority.
      R_G/C_G when loaded: [1,H,W] fused/canonical geometry teacher.
      D_teacher/C_teacher are backward-compatible [1,H/4,W/4] aliases.
      D_da_raw/da_raw_valid when loaded: [1,H,W] dense relative mono teacher.
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
        load_geometry: bool = False,
        load_mono: bool = False,
        mono_key: str = "D_da_raw",
        min_depth: float = 1e-3,
        max_depth: float = 120.0,
        calibrate_metric_teacher: bool = True,
        metric_conf_min: float = 0.05,
        metric_conf_sparse_decay: float = 6.0,
        metric_conf_range_decay: float = 0.25,
        metric_conf_sparse_blend_radius: float = 48.0,
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
        self.load_geometry = load_geometry
        self.load_mono = load_mono
        self.mono_key = mono_key
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.calibrate_metric_teacher = bool(calibrate_metric_teacher)
        self.metric_conf_min = float(metric_conf_min)
        self.metric_conf_sparse_decay = float(metric_conf_sparse_decay)
        self.metric_conf_range_decay = float(metric_conf_range_decay)
        self.metric_conf_sparse_blend_radius = float(metric_conf_sparse_blend_radius)
        self.return_tensors = return_tensors
        self._warned_dmd_geometry_fallback = False

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
            D_cm, C_cm = self._load_metric_teacher(
                sample["sample_id"],
                sample["rgb"].shape[:2],
                sample["gt"],
                sample["gt_mask"],
                sample["sparse"],
                sample["mask"],
            )
            out["D_cm"] = torch.from_numpy(D_cm[None]).float()
            out["C_cm"] = torch.from_numpy(C_cm[None]).float()
            D_teacher = _downsample_map(D_cm, self.output_scale, interpolation=cv2.INTER_AREA)
            C_teacher = _downsample_map(C_cm, self.output_scale, interpolation=cv2.INTER_AREA)
            out["D_teacher"] = torch.from_numpy(D_teacher[None]).float()
            out["C_teacher"] = torch.from_numpy(C_teacher[None]).float()
        if self.load_geometry:
            R_G, C_G = self._load_geometry_teacher(sample["sample_id"], sample["rgb"].shape[:2])
            out["R_G"] = torch.from_numpy(R_G[None]).float()
            out["C_G"] = torch.from_numpy(C_G[None]).float()
        if self.load_mono:
            D_da_raw, da_raw_valid = self._load_da_raw(sample["sample_id"], sample["rgb"].shape[:2])
            out["D_da_raw"] = torch.from_numpy(D_da_raw[None]).float()
            out["da_raw_valid"] = torch.from_numpy(da_raw_valid[None]).float()
        return out

    def _load_metric_teacher(
        self,
        sample_id: str,
        image_hw: tuple[int, int],
        gt: np.ndarray,
        gt_mask: np.ndarray,
        sparse: np.ndarray,
        sparse_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load the theory metric branch: D_cm/C_cm = GT priority, otherwise DMD3C."""
        h, w = image_hw
        zeros = np.zeros((h, w), dtype=np.float32)
        if self.teacher_root is None:
            return np.where(gt_mask > 0.5, gt, zeros).astype(np.float32), (gt_mask > 0.5).astype(np.float32)

        D_cm: np.ndarray | None = None
        C_cm: np.ndarray | None = None
        candidates = [
            (
                self.teacher_root / "metric_coarse" / self.split_name / f"{sample_id}.npz",
                ("D_cm", "D_full", "D_teacher"),
                ("C_cm", "C_full", "C_teacher", "C_dmd3c"),
            ),
            (
                self.teacher_root / "fused" / self.split_name / f"{sample_id}.npz",
                ("D_cm", "D_full", "D_teacher"),
                ("C_cm", "C_full", "C_teacher", "C_dmd3c"),
            ),
        ]
        for path, depth_keys, conf_keys in candidates:
            depth = _load_first_npz_key(path, depth_keys, image_hw, cv2.INTER_LINEAR)
            if depth is None:
                continue
            conf = _load_first_npz_key(path, conf_keys, image_hw, cv2.INTER_LINEAR)
            D_cm = depth
            C_cm = conf if conf is not None else _valid_depth(depth, self.min_depth, self.max_depth).astype(np.float32)
            break

        if D_cm is None:
            dmd_path = self.teacher_root / "dmd3c" / self.split_name / f"{sample_id}.npz"
            D_dmd = _load_first_npz_key(dmd_path, ("D_dmd3c", "D_dmd", "D"), image_hw, cv2.INTER_LINEAR)
            if D_dmd is not None:
                D_cm = D_dmd
                C_cm = None

        if D_cm is None:
            D_cm = zeros.copy()
            C_cm = zeros.copy()
        else:
            gt_valid = (gt_mask > 0.5) & _valid_depth(gt, self.min_depth, self.max_depth)
            if self.calibrate_metric_teacher:
                D_cm = _calibrate_to_gt(D_cm, gt, gt_valid, self.min_depth, self.max_depth)
            valid = _valid_depth(D_cm, self.min_depth, self.max_depth)
            D_cm = np.where(valid, D_cm, 0.0).astype(np.float32)
            auto_conf = _metric_confidence(
                D_cm,
                sparse,
                sparse_mask,
                self.min_depth,
                self.max_depth,
                self.metric_conf_min,
                self.metric_conf_sparse_decay,
                self.metric_conf_range_decay,
                self.metric_conf_sparse_blend_radius,
            )
            if C_cm is None:
                C_cm = auto_conf
            else:
                C_cm = np.clip(np.where(np.isfinite(C_cm), C_cm, 0.0), 0.0, 1.0).astype(np.float32)
                C_cm = (C_cm * auto_conf).astype(np.float32)
            C_cm[~valid] = 0.0

        gt_valid = (gt_mask > 0.5) & _valid_depth(gt, self.min_depth, self.max_depth)
        D_cm[gt_valid] = gt[gt_valid].astype(np.float32)
        C_cm[gt_valid] = 1.0
        return D_cm.astype(np.float32), C_cm.astype(np.float32)

    def _load_geometry_teacher(self, sample_id: str, image_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """Load R_G/C_G. Falls back to DA raw as structure-only supervision."""
        h, w = image_hw
        zeros = np.zeros((h, w), dtype=np.float32)
        if self.teacher_root is None:
            return zeros, zeros

        R: np.ndarray | None = None
        C: np.ndarray | None = None
        valid: np.ndarray | None = None
        source: str | None = None

        path = self.teacher_root / "geometry_fused" / self.split_name / f"{sample_id}.npz"
        R = _load_first_npz_key(path, ("R_G", "R_G_star", "R_fused", "R_teacher"), image_hw, cv2.INTER_LINEAR)
        if R is not None:
            C = _load_first_npz_key(path, ("C_G", "C_geometry", "C_teacher"), image_hw, cv2.INTER_LINEAR)
            valid = np.isfinite(R)
            source = "geometry_fused"

        if R is None:
            fallbacks = [
                (Path("geometry_raw") / "depth_anything_v2" / self.split_name, ("R_da", "D_da_raw", "R_i", "depth"), "raw", "depth_anything_v2"),
                (Path("geometry_raw") / "depth_anything" / self.split_name, ("R_da", "D_da_raw", "R_i", "depth"), "raw", "depth_anything"),
                (Path("depth_anything") / f"{self.split_name}_raw", ("D_da_raw", "R_i", "depth"), "raw", "depth_anything_raw"),
                (Path("depth_anything") / self.split_name, ("D_da_raw", "R_i", "depth"), "raw", "depth_anything"),
                (Path("depth_anything") / f"{self.split_name}_aligned", ("D_da_aligned", "D_full", "D_teacher"), "log_metric", "depth_anything_aligned"),
                (Path("metric3d") / self.split_name, ("D_m3d",), "log_metric", "metric3d"),
                (Path("dmd3c") / self.split_name, ("D_dmd3c", "D_dmd", "D"), "log_metric", "dmd3c"),
            ]
            for rel, keys, transform, candidate_source in fallbacks:
                arr = _load_first_npz_key(self.teacher_root / rel / f"{sample_id}.npz", keys, image_hw, cv2.INTER_LINEAR)
                if arr is None:
                    continue
                if transform == "log_metric":
                    valid = _valid_depth(arr, self.min_depth, self.max_depth)
                    R = np.log(np.clip(arr, self.min_depth, self.max_depth))
                else:
                    valid = np.isfinite(arr)
                    R = arr
                C = valid.astype(np.float32)
                source = candidate_source
                break

        if R is None:
            return zeros, zeros
        if source == "dmd3c" and not self._warned_dmd_geometry_fallback:
            LOGGER.warning(
                "Main geometry teachers are missing for split=%s; using DMD3C-only geometry fallback for at least one sample.",
                self.split_name,
            )
            self._warned_dmd_geometry_fallback = True

        R = _robust_normalize(R, valid)
        if C is None:
            C = np.isfinite(R).astype(np.float32)
        C = np.clip(np.where(np.isfinite(C), C, 0.0), 0.0, 1.0).astype(np.float32)
        C[~np.isfinite(R)] = 0.0
        return R.astype(np.float32), C.astype(np.float32)

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

    def _load_da_raw(self, sample_id: str, image_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """Load Depth Anything raw relative depth for SSI distillation.

        Supported legacy locations:
          teacher_outputs/depth_anything/{split}_raw/{sample_id}.npz
          teacher_outputs/depth_anything/{split}/{sample_id}.npz
          teacher_outputs/depth_anything/{split}_aligned/{sample_id}.npz

        Expected NPZ key:
          D_da_raw: float32 [H,W], relative monocular depth.
        """
        h, w = image_hw
        if self.teacher_root is None:
            zeros = np.zeros((h, w), dtype=np.float32)
            return zeros, zeros

        root = self.teacher_root / "depth_anything"
        candidates = [
            self.teacher_root / "geometry_raw" / "depth_anything_v2" / self.split_name / f"{sample_id}.npz",
            self.teacher_root / "geometry_raw" / "depth_anything" / self.split_name / f"{sample_id}.npz",
            root / f"{self.split_name}_raw" / f"{sample_id}.npz",
            root / self.split_name / f"{sample_id}.npz",
            root / f"{self.split_name}_aligned" / f"{sample_id}.npz",
        ]
        for path in candidates:
            if not path.exists():
                continue
            try:
                with np.load(path) as data:
                    key = next((k for k in (self.mono_key, "R_da", "D_da_raw", "R_i", "depth", "D_da_aligned", "D_full") if k in data), None)
                    if key is None:
                        continue
                    raw = data[key].astype(np.float32)
            except Exception:
                continue
            if raw.ndim == 3:
                if raw.shape[0] == 1:
                    raw = raw[0]
                elif raw.shape[-1] == 1:
                    raw = raw[..., 0]
                else:
                    raw = raw[..., 0]
            if raw.shape != (h, w):
                raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            valid = np.isfinite(raw).astype(np.float32)
            raw = np.where(np.isfinite(raw), raw, 0.0).astype(np.float32)
            return raw, valid

        zeros = np.zeros((h, w), dtype=np.float32)
        return zeros, zeros

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
