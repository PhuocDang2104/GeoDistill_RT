from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np

from .utils import make_ray_map, make_uv_map


@dataclass(frozen=True)
class DepthCompletionAugmentationConfig:
    """Train-only augmentation contract for aligned depth-completion fields."""

    enabled: bool = False
    stage: str = "none"
    horizontal_flip_prob: float = 0.0
    sparse_dropout_min_rate: float = 0.0
    sparse_dropout_max_rate: float = 0.0
    sparse_dropout_min_points: int = 64
    scale_min: float = 1.0
    scale_max: float = 1.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "DepthCompletionAugmentationConfig":
        cfg = dict(value or {})
        dropout = dict(cfg.get("sparse_dropout", {}) or {})
        scale = dict(cfg.get("scale_jitter", {}) or {})
        result = cls(
            enabled=bool(cfg.get("enabled", False)),
            stage=str(cfg.get("stage", "custom" if cfg.get("enabled", False) else "none")),
            horizontal_flip_prob=float(cfg.get("horizontal_flip_prob", 0.0)),
            sparse_dropout_min_rate=(
                float(dropout.get("min_rate", 0.0)) if bool(dropout.get("enabled", False)) else 0.0
            ),
            sparse_dropout_max_rate=(
                float(dropout.get("max_rate", 0.0)) if bool(dropout.get("enabled", False)) else 0.0
            ),
            sparse_dropout_min_points=int(dropout.get("min_points", 64)),
            scale_min=float(scale.get("min_scale", 1.0)) if bool(scale.get("enabled", False)) else 1.0,
            scale_max=float(scale.get("max_scale", 1.0)) if bool(scale.get("enabled", False)) else 1.0,
        )
        result.validate()
        return result

    def validate(self) -> None:
        if not 0.0 <= self.horizontal_flip_prob <= 1.0:
            raise ValueError("horizontal_flip_prob must be in [0, 1]")
        if not 0.0 <= self.sparse_dropout_min_rate <= self.sparse_dropout_max_rate < 1.0:
            raise ValueError("sparse dropout rates must satisfy 0 <= min <= max < 1")
        if self.sparse_dropout_min_points < 0:
            raise ValueError("sparse_dropout_min_points must be non-negative")
        if not 1.0 <= self.scale_min <= self.scale_max:
            raise ValueError("The current crop-only scale jitter requires 1 <= min_scale <= max_scale")


def _randint(rng: Any, low: int, high_exclusive: int) -> int:
    if high_exclusive <= low:
        return low
    if hasattr(rng, "integers"):
        return int(rng.integers(low, high_exclusive))
    return int(rng.randint(low, high_exclusive))


def _resize_crop(
    array: np.ndarray,
    resized_hw: tuple[int, int],
    crop_xy: tuple[int, int],
    output_hw: tuple[int, int],
    interpolation: int,
) -> np.ndarray:
    out_h, out_w = output_hw
    crop_x, crop_y = crop_xy
    resized = cv2.resize(array, (resized_hw[1], resized_hw[0]), interpolation=interpolation)
    return np.ascontiguousarray(resized[crop_y : crop_y + out_h, crop_x : crop_x + out_w])


def _resize_crop_weighted(
    value: np.ndarray,
    confidence: np.ndarray,
    resized_hw: tuple[int, int],
    crop_xy: tuple[int, int],
    output_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    weight = np.clip(np.where(np.isfinite(confidence), confidence, 0.0), 0.0, 1.0).astype(np.float32)
    clean = np.where(np.isfinite(value), value, 0.0).astype(np.float32)
    numerator = _resize_crop(clean * weight, resized_hw, crop_xy, output_hw, cv2.INTER_LINEAR)
    denominator = _resize_crop(weight, resized_hw, crop_xy, output_hw, cv2.INTER_LINEAR)
    resized_value = np.zeros_like(numerator, dtype=np.float32)
    np.divide(numerator, denominator, out=resized_value, where=denominator > 1e-6)
    return resized_value, np.clip(denominator, 0.0, 1.0).astype(np.float32)


def _warp_sparse_points(
    sparse: np.ndarray,
    resized_hw: tuple[int, int],
    crop_xy: tuple[int, int],
    output_hw: tuple[int, int],
) -> np.ndarray:
    """Forward-warp each LiDAR sample once; never duplicate a point during zoom."""

    in_h, in_w = sparse.shape
    resized_h, resized_w = resized_hw
    out_h, out_w = output_hw
    crop_x, crop_y = crop_xy
    ys, xs = np.nonzero(np.isfinite(sparse) & (sparse > 0.0))
    output = np.zeros((out_h, out_w), dtype=np.float32)
    if ys.size == 0:
        return output

    scale_x = resized_w / float(in_w)
    scale_y = resized_h / float(in_h)
    xs_new = np.rint((xs.astype(np.float64) + 0.5) * scale_x - 0.5).astype(np.int64) - crop_x
    ys_new = np.rint((ys.astype(np.float64) + 0.5) * scale_y - 0.5).astype(np.int64) - crop_y
    valid = (xs_new >= 0) & (xs_new < out_w) & (ys_new >= 0) & (ys_new < out_h)
    if not np.any(valid):
        return output

    flat_indices = ys_new[valid] * out_w + xs_new[valid]
    flat_depth = np.full(out_h * out_w, np.inf, dtype=np.float32)
    # For collisions, retain the surface nearest to the camera.
    np.minimum.at(flat_depth, flat_indices, sparse[ys[valid], xs[valid]].astype(np.float32))
    flat_depth[~np.isfinite(flat_depth)] = 0.0
    return flat_depth.reshape(out_h, out_w)


def _apply_sparse_dropout(
    sparse: np.ndarray,
    min_rate: float,
    max_rate: float,
    min_points: int,
    rng: Any,
) -> tuple[np.ndarray, float, int, int]:
    valid_flat = np.flatnonzero(np.isfinite(sparse.reshape(-1)) & (sparse.reshape(-1) > 0.0))
    points_before = int(valid_flat.size)
    if points_before == 0 or max_rate <= 0.0:
        return sparse.astype(np.float32, copy=True), 0.0, points_before, points_before

    rate = float(rng.uniform(min_rate, max_rate)) if max_rate > min_rate else float(max_rate)
    keep = np.asarray(rng.random(points_before)) >= rate
    required = min(points_before, int(min_points))
    if int(keep.sum()) < required:
        keep[:] = False
        selected = np.asarray(rng.choice(points_before, size=required, replace=False), dtype=np.int64)
        keep[selected] = True

    flat = sparse.astype(np.float32, copy=True).reshape(-1)
    flat[valid_flat[~keep]] = 0.0
    return flat.reshape(sparse.shape), rate, points_before, int(keep.sum())


class DepthCompletionAugmentor:
    """Apply one shared spatial transform to inputs, labels, teachers and camera geometry."""

    _WEIGHTED_TEACHER_PAIRS = (
        ("D_cm", "C_cm"),
        ("R_G", "C_G"),
        ("D_da_raw", "da_raw_valid"),
    )
    _FLIP_KEYS = (
        "rgb",
        "sparse",
        "gt",
        "mask",
        "gt_mask",
        "D_cm",
        "C_cm",
        "R_G",
        "C_G",
        "D_da_raw",
        "da_raw_valid",
    )

    def __init__(self, config: Mapping[str, Any] | DepthCompletionAugmentationConfig | None) -> None:
        self.config = (
            config if isinstance(config, DepthCompletionAugmentationConfig) else DepthCompletionAugmentationConfig.from_mapping(config)
        )

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def __call__(self, sample: dict[str, Any], rng: Any | None = None) -> dict[str, Any]:
        if not self.enabled:
            return sample
        rng = np.random if rng is None else rng
        output = dict(sample)
        rgb = np.asarray(output["rgb"])
        out_h, out_w = rgb.shape[:2]
        scale = float(rng.uniform(self.config.scale_min, self.config.scale_max))
        resized_h = max(out_h, int(round(out_h * scale)))
        resized_w = max(out_w, int(round(out_w * scale)))
        scale_y = resized_h / float(out_h)
        scale_x = resized_w / float(out_w)
        crop_x = _randint(rng, 0, resized_w - out_w + 1)
        crop_y = _randint(rng, 0, resized_h - out_h + 1)

        if resized_h != out_h or resized_w != out_w or crop_x or crop_y:
            output["rgb"] = _resize_crop(rgb, (resized_h, resized_w), (crop_x, crop_y), (out_h, out_w), cv2.INTER_LINEAR)
            output["sparse"] = _warp_sparse_points(
                np.asarray(output["sparse"]), (resized_h, resized_w), (crop_x, crop_y), (out_h, out_w)
            )
            output["gt"] = _resize_crop(
                np.asarray(output["gt"]), (resized_h, resized_w), (crop_x, crop_y), (out_h, out_w), cv2.INTER_NEAREST
            ).astype(np.float32)
            handled: set[str] = set()
            for value_key, confidence_key in self._WEIGHTED_TEACHER_PAIRS:
                if value_key in output and confidence_key in output:
                    output[value_key], output[confidence_key] = _resize_crop_weighted(
                        np.asarray(output[value_key]),
                        np.asarray(output[confidence_key]),
                        (resized_h, resized_w),
                        (crop_x, crop_y),
                        (out_h, out_w),
                    )
                    handled.update((value_key, confidence_key))
            for key in ("D_cm", "C_cm", "R_G", "C_G", "D_da_raw", "da_raw_valid"):
                if key in output and key not in handled:
                    output[key] = _resize_crop(
                        np.asarray(output[key]),
                        (resized_h, resized_w),
                        (crop_x, crop_y),
                        (out_h, out_w),
                        cv2.INTER_LINEAR,
                    ).astype(np.float32)

            K = np.asarray(output["K"], dtype=np.float32).copy()
            K[0, 0] *= scale_x
            K[1, 1] *= scale_y
            K[0, 2] = (K[0, 2] + 0.5) * scale_x - 0.5 - crop_x
            K[1, 2] = (K[1, 2] + 0.5) * scale_y - 0.5 - crop_y
            output["K"] = K

        flipped = bool(float(rng.random()) < self.config.horizontal_flip_prob)
        if flipped:
            for key in self._FLIP_KEYS:
                if key in output:
                    output[key] = np.ascontiguousarray(np.asarray(output[key])[:, ::-1])
            K = np.asarray(output["K"], dtype=np.float32).copy()
            K[0, 2] = float(out_w - 1) - K[0, 2]
            output["K"] = K

        sparse_aug, dropout_rate, points_before, points_after = _apply_sparse_dropout(
            np.asarray(output["sparse"]),
            self.config.sparse_dropout_min_rate,
            self.config.sparse_dropout_max_rate,
            self.config.sparse_dropout_min_points,
            rng,
        )
        output["sparse"] = sparse_aug
        output["mask"] = ((sparse_aug > 0.0) & np.isfinite(sparse_aug)).astype(np.float32)
        gt = np.asarray(output["gt"], dtype=np.float32)
        output["gt_mask"] = ((gt > 0.0) & np.isfinite(gt)).astype(np.float32)
        # Preserve the dataset's GT-priority metric-teacher contract after
        # bilinear teacher warping and nearest-neighbor GT warping.
        if "D_cm" in output and "C_cm" in output:
            gt_valid = output["gt_mask"] > 0.5
            metric_teacher = np.asarray(output["D_cm"], dtype=np.float32).copy()
            metric_confidence = np.asarray(output["C_cm"], dtype=np.float32).copy()
            metric_teacher[gt_valid] = gt[gt_valid]
            metric_confidence[gt_valid] = 1.0
            output["D_cm"], output["C_cm"] = metric_teacher, metric_confidence
        for key in ("C_cm", "C_G", "da_raw_valid"):
            if key in output:
                output[key] = np.clip(np.asarray(output[key]), 0.0, 1.0).astype(np.float32)

        K = np.asarray(output["K"], dtype=np.float32)
        output["ray"] = make_ray_map(K, out_h, out_w)
        output["uv"] = make_uv_map(out_h, out_w)
        output["augmentation_params"] = np.asarray(
            [scale_x, scale_y, crop_x, crop_y, float(flipped), dropout_rate, points_before, points_after],
            dtype=np.float32,
        )
        return output
