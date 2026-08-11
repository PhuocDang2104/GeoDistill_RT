from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import torch


def valid_mask(depth: torch.Tensor, min_depth: float = 1e-3, max_depth: float = 120.0) -> torch.Tensor:
    return torch.isfinite(depth) & (depth > min_depth) & (depth < max_depth)


def masked_mean(value: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mask_f = mask.to(dtype=value.dtype)
    return (value * mask_f).sum() / mask_f.sum().clamp_min(eps)


class GlobalDepthMetricAccumulator:
    """Accumulate KITTI depth metrics over all valid pixels, not per-image means."""

    def __init__(self, min_depth: float = 1e-3, max_depth: float = 120.0) -> None:
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.sq = 0.0
        self.abs = 0.0
        self.inv_sq = 0.0
        self.inv_abs = 0.0
        self.abs_rel = 0.0
        self.delta1 = 0.0
        self.delta2 = 0.0
        self.delta3 = 0.0
        self.count = 0

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        pred = pred.detach().float()
        target = target.detach().float()
        valid = valid_mask(target, self.min_depth, self.max_depth)
        if mask is not None:
            valid = valid & mask.detach().bool()
        count = int(valid.sum().item())
        if count < 1:
            return
        prediction = pred.clamp(self.min_depth, self.max_depth)[valid]
        truth = target.clamp(self.min_depth, self.max_depth)[valid]
        diff = prediction - truth
        inv_diff = 1000.0 / prediction - 1000.0 / truth
        ratio = torch.maximum(prediction / truth, truth / prediction)
        self.sq += float((diff * diff).sum().cpu())
        self.abs += float(diff.abs().sum().cpu())
        self.inv_sq += float((inv_diff * inv_diff).sum().cpu())
        self.inv_abs += float(inv_diff.abs().sum().cpu())
        self.abs_rel += float((diff.abs() / truth).sum().cpu())
        self.delta1 += float((ratio < 1.25).sum().cpu())
        self.delta2 += float((ratio < 1.25**2).sum().cpu())
        self.delta3 += float((ratio < 1.25**3).sum().cpu())
        self.count += count

    def compute(self) -> dict[str, float]:
        if self.count < 1:
            return {
                "rmse": float("inf"),
                "mae": float("inf"),
                "irmse": float("inf"),
                "imae": float("inf"),
                "abs_rel": float("inf"),
                "delta1": 0.0,
                "delta2": 0.0,
                "delta3": 0.0,
                "valid_pixels": 0,
            }
        count = float(self.count)
        return {
            "rmse": math.sqrt(self.sq / count),
            "mae": self.abs / count,
            "irmse": math.sqrt(self.inv_sq / count),
            "imae": self.inv_abs / count,
            "abs_rel": self.abs_rel / count,
            "delta1": self.delta1 / count,
            "delta2": self.delta2 / count,
            "delta3": self.delta3 / count,
            "valid_pixels": self.count,
        }


def depth_metrics_torch(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    min_depth: float = 1e-3,
    max_depth: float = 120.0,
) -> dict[str, float]:
    """Compute depth metrics with tensors shaped [B,1,H,W] or [B,H,W]."""
    pred = pred.float()
    target = target.float()
    if mask is None:
        mask = valid_mask(target, min_depth, max_depth)
    else:
        mask = mask.bool() & valid_mask(target, min_depth, max_depth)
    pred = pred.clamp_min(min_depth)
    target = target.clamp_min(min_depth)
    diff = pred - target
    abs_diff = diff.abs()
    rmse = torch.sqrt(masked_mean(diff * diff, mask)).item()
    mae = masked_mean(abs_diff, mask).item()
    abs_rel = masked_mean(abs_diff / target, mask).item()
    # KITTI convention reports inverse-depth errors in 1/km while depth is in metres.
    inv_diff = 1000.0 / pred - 1000.0 / target
    irmse = torch.sqrt(masked_mean(inv_diff * inv_diff, mask)).item()
    imae = masked_mean(inv_diff.abs(), mask).item()
    ratio = torch.maximum(pred / target, target / pred)
    a1 = masked_mean((ratio < 1.25).float(), mask).item()
    a2 = masked_mean((ratio < 1.25**2).float(), mask).item()
    a3 = masked_mean((ratio < 1.25**3).float(), mask).item()
    return {
        "rmse": rmse,
        "mae": mae,
        "irmse": irmse,
        "imae": imae,
        "abs_rel": abs_rel,
        "delta1": a1,
        "delta2": a2,
        "delta3": a3,
    }


def _range_key(lo: float, hi: float) -> str:
    return f"{int(lo)}_{int(hi)}"


def depth_metrics_by_range_torch(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    bins: list[float] | tuple[float, ...] = (0.0, 20.0, 40.0, 60.0, 80.0, 120.0),
    min_depth: float = 1e-3,
    max_depth: float = 120.0,
) -> dict[str, float]:
    out: dict[str, float] = {}
    base_mask = mask.bool() & valid_mask(target, min_depth, max_depth)
    diff = pred.float().clamp_min(min_depth) - target.float().clamp_min(min_depth)
    abs_diff = diff.abs()
    for idx in range(len(bins) - 1):
        lo = float(bins[idx])
        hi = float(bins[idx + 1])
        m = base_mask & (target >= lo) & (target < hi)
        if int(m.sum().item()) < 1:
            continue
        key = _range_key(lo, hi)
        out[f"rmse_{key}"] = torch.sqrt(masked_mean(diff * diff, m)).item()
        out[f"mae_{key}"] = masked_mean(abs_diff, m).item()
    return out


def depth_metrics_by_edge_torch(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    rgb: torch.Tensor,
    edge_threshold: float = 0.05,
    min_depth: float = 1e-3,
    max_depth: float = 120.0,
) -> dict[str, float]:
    base_mask = mask.bool() & valid_mask(target, min_depth, max_depth)
    gray = rgb.float().mean(dim=1, keepdim=True)
    gx = torch.zeros_like(gray)
    gy = torch.zeros_like(gray)
    gx[..., :, 1:] = (gray[..., :, 1:] - gray[..., :, :-1]).abs()
    gy[..., 1:, :] = (gray[..., 1:, :] - gray[..., :-1, :]).abs()
    edge = (torch.maximum(gx, gy) > edge_threshold) & base_mask
    nonedge = (~edge) & base_mask
    diff = pred.float().clamp_min(min_depth) - target.float().clamp_min(min_depth)
    out: dict[str, float] = {}
    if int(edge.sum().item()) > 0:
        out["rmse_edge"] = torch.sqrt(masked_mean(diff * diff, edge)).item()
        out["mae_edge"] = masked_mean(diff.abs(), edge).item()
    if int(nonedge.sum().item()) > 0:
        out["rmse_nonedge"] = torch.sqrt(masked_mean(diff * diff, nonedge)).item()
        out["mae_nonedge"] = masked_mean(diff.abs(), nonedge).item()
    return out


def depth_metrics_np(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
    min_depth: float = 1e-3,
    max_depth: float = 120.0,
) -> dict[str, float]:
    pred_t = torch.from_numpy(pred).float()
    target_t = torch.from_numpy(target).float()
    mask_t = torch.from_numpy(mask.astype(bool)) if mask is not None else None
    return depth_metrics_torch(pred_t, target_t, mask_t, min_depth, max_depth)


class AverageMeter:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        if math.isfinite(float(value)):
            self.sum += float(value) * n
            self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)


def average_metric_dict(records: list[Mapping[str, float]]) -> dict[str, float]:
    meters: dict[str, AverageMeter] = {}
    for record in records:
        for key, value in record.items():
            meters.setdefault(key, AverageMeter()).update(float(value))
    return {key: meter.avg for key, meter in meters.items()}
