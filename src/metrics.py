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
    ratio = torch.maximum(pred / target, target / pred)
    a1 = masked_mean((ratio < 1.25).float(), mask).item()
    a2 = masked_mean((ratio < 1.25**2).float(), mask).item()
    a3 = masked_mean((ratio < 1.25**3).float(), mask).item()
    return {"rmse": rmse, "mae": mae, "abs_rel": abs_rel, "delta1": a1, "delta2": a2, "delta3": a3}


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
