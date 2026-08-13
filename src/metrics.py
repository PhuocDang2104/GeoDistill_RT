from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F


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


class DepthPredictionAudit:
    """Streaming near-zero and inverse-depth outlier audit on the GT grid."""

    def __init__(
        self,
        min_depth: float = 1e-3,
        max_depth: float = 120.0,
        protocol_min_depth: float | None = None,
        protocol_max_depth: float | None = None,
        stages: tuple[str, ...] = ("D8", "D4", "D2", "D1", "D_full"),
        thresholds: tuple[float, ...] = (0.01, 0.05, 0.1, 0.5),
        top_k: int = 100,
        histogram_bins: int = 8192,
    ) -> None:
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.protocol_min_depth = float(protocol_min_depth if protocol_min_depth is not None else min_depth)
        self.protocol_max_depth = float(protocol_max_depth if protocol_max_depth is not None else max_depth)
        self.stages = tuple(stages)
        self.thresholds = tuple(float(value) for value in thresholds)
        self.top_k = int(top_k)
        self.histogram_bins = int(histogram_bins)
        self.hist_log_min = math.log(1e-6)
        self.hist_log_max = math.log(max(self.max_depth, self.protocol_max_depth, 1.0))
        self.stage_stats: dict[str, dict[str, Any]] = {
            stage: {
                "min": float("inf"),
                "count": 0,
                "nonpositive_or_nonfinite": 0,
                "low_counts": {threshold: 0 for threshold in self.thresholds},
                "hist": torch.zeros(self.histogram_bins, dtype=torch.float64),
                "sum": 0.0,
            }
            for stage in self.stages
        }
        self.inverse_stats = {
            stage: {"raw_sq": 0.0, "raw_count": 0, "raw_invalid": 0, "protocol_sq": 0.0, "protocol_count": 0}
            for stage in ("D1", "D_full")
        }
        self.top_records: list[dict[str, Any]] = []

    @staticmethod
    def _sample_ids(batch: dict[str, Any], batch_size: int) -> list[str]:
        raw = batch.get("sample_id", [])
        if isinstance(raw, (list, tuple)):
            return [str(value) for value in raw]
        return [str(raw)] * batch_size

    def _to_gt_grid(self, value: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        value = value.detach().float()
        if value.shape[-2:] == target_hw:
            return value
        return F.interpolate(value, size=target_hw, mode="bilinear", align_corners=False)

    def update(self, predictions: dict[str, torch.Tensor], batch: dict[str, Any]) -> None:
        target = batch["gt"].detach().float()
        target_mask = (
            (batch["gt_mask"] > 0.5)
            & torch.isfinite(target)
            & (target > self.min_depth)
            & (target < self.max_depth)
        )
        if int(target_mask.sum().item()) < 1:
            return
        aligned: dict[str, torch.Tensor] = {}
        for stage in self.stages:
            if stage not in predictions:
                continue
            value = self._to_gt_grid(predictions[stage], target.shape[-2:])
            aligned[stage] = value
            values = value[target_mask]
            finite_positive = torch.isfinite(values) & (values > 0.0)
            valid_values = values[finite_positive]
            stats = self.stage_stats[stage]
            stats["count"] += int(values.numel())
            stats["nonpositive_or_nonfinite"] += int((~finite_positive).sum().item())
            if valid_values.numel() > 0:
                stats["min"] = min(float(stats["min"]), float(valid_values.min().item()))
                stats["sum"] += float(valid_values.sum().cpu())
                for threshold in self.thresholds:
                    stats["low_counts"][threshold] += int((valid_values < threshold).sum().item())
                log_values = valid_values.clamp_min(1e-6).log()
                hist = torch.histc(
                    log_values,
                    bins=self.histogram_bins,
                    min=self.hist_log_min,
                    max=self.hist_log_max,
                )
                stats["hist"] += hist.double().cpu()

        for stage in ("D1", "D_full"):
            if stage not in aligned:
                continue
            prediction = aligned[stage]
            raw_valid = target_mask & torch.isfinite(prediction) & (prediction > 0.0)
            raw_invalid = target_mask & ~raw_valid
            inv = self.inverse_stats[stage]
            inv["raw_invalid"] += int(raw_invalid.sum().item())
            if int(raw_valid.sum().item()) > 0:
                raw_diff = 1000.0 / prediction[raw_valid] - 1000.0 / target[raw_valid]
                inv["raw_sq"] += float((raw_diff * raw_diff).sum().cpu())
                inv["raw_count"] += int(raw_valid.sum().item())
            protocol_prediction = prediction.clamp(self.protocol_min_depth, self.protocol_max_depth)
            protocol_target = target.clamp(self.protocol_min_depth, self.protocol_max_depth)
            protocol_diff = 1000.0 / protocol_prediction[target_mask] - 1000.0 / protocol_target[target_mask]
            inv["protocol_sq"] += float((protocol_diff * protocol_diff).sum().cpu())
            inv["protocol_count"] += int(target_mask.sum().item())

        if "D1" not in aligned:
            return
        ranking_prediction = aligned["D1"]
        ranking_valid = target_mask & torch.isfinite(ranking_prediction) & (ranking_prediction > 0.0)
        if int(ranking_valid.sum().item()) < 1:
            return
        error_map = torch.full_like(target, float("-inf"))
        error_map[ranking_valid] = (
            1000.0 / ranking_prediction[ranking_valid] - 1000.0 / target[ranking_valid]
        ).abs()
        flat = error_map.flatten()
        count = min(self.top_k, int(ranking_valid.sum().item()))
        errors, indices = torch.topk(flat, k=count)
        height, width = target.shape[-2:]
        image_area = height * width
        batch_indices = indices // image_area
        within = indices % image_area
        ys = within // width
        xs = within % width
        sample_ids = self._sample_ids(batch, target.shape[0])
        stage_cpu = {
            stage: value.flatten()[indices].detach().cpu().tolist()
            for stage, value in aligned.items()
        }
        gt_values = target.flatten()[indices].detach().cpu().tolist()
        errors_cpu = errors.detach().cpu().tolist()
        batch_cpu = batch_indices.detach().cpu().tolist()
        ys_cpu = ys.detach().cpu().tolist()
        xs_cpu = xs.detach().cpu().tolist()
        for index in range(count):
            self.top_records.append(
                {
                    "inverse_error_1_per_km": float(errors_cpu[index]),
                    "sample_id": sample_ids[int(batch_cpu[index])],
                    "u": int(xs_cpu[index]),
                    "v": int(ys_cpu[index]),
                    "gt_m": float(gt_values[index]),
                    "stage_prediction_m": {
                        stage: float(values[index]) for stage, values in stage_cpu.items()
                    },
                }
            )
        self.top_records.sort(key=lambda row: row["inverse_error_1_per_km"], reverse=True)
        del self.top_records[self.top_k :]

    def _quantile(self, histogram: torch.Tensor, probability: float) -> float:
        count = float(histogram.sum().item())
        if count < 1:
            return float("nan")
        target = max(1.0, probability * count)
        index = int(torch.searchsorted(histogram.cumsum(0), torch.tensor(target, dtype=torch.float64)).item())
        index = min(max(index, 0), self.histogram_bins - 1)
        log_value = self.hist_log_min + (index + 0.5) * (self.hist_log_max - self.hist_log_min) / self.histogram_bins
        return math.exp(log_value)

    def report(self) -> dict[str, Any]:
        quantiles = {"q0001": 0.0001, "q001": 0.001, "q01": 0.01, "q1": 0.1}
        stages: dict[str, Any] = {}
        for stage, stats in self.stage_stats.items():
            stages[stage] = {
                "pred_min_m": float(stats["min"]) if math.isfinite(float(stats["min"])) else None,
                "pred_mean_m": float(stats["sum"]) / max(1, int(stats["count"]) - int(stats["nonpositive_or_nonfinite"])),
                **{
                    f"pred_{name}_m": self._quantile(stats["hist"], probability)
                    for name, probability in quantiles.items()
                },
                "valid_gt_pixels": int(stats["count"]),
                "nonpositive_or_nonfinite": int(stats["nonpositive_or_nonfinite"]),
                "low_depth_counts": {f"lt_{threshold:g}m": int(stats["low_counts"][threshold]) for threshold in self.thresholds},
            }
        inverse: dict[str, Any] = {}
        for stage, stats in self.inverse_stats.items():
            inverse[stage] = {
                "irmse_raw_1_per_km": math.sqrt(stats["raw_sq"] / max(1, stats["raw_count"])),
                "irmse_protocol_1_per_km": math.sqrt(stats["protocol_sq"] / max(1, stats["protocol_count"])),
                "raw_valid_pixels": int(stats["raw_count"]),
                "raw_invalid_pixels": int(stats["raw_invalid"]),
            }
        return {
            "scope": "Predictions bilinearly aligned to the full GT grid; statistics use valid GT pixels only.",
            "quantile_probabilities": quantiles,
            "quantile_note": "Quantiles are streaming log-histogram estimates; min/count/iRMSE/outlier values are exact.",
            "protocol_bounds_m": [self.protocol_min_depth, self.protocol_max_depth],
            "stages": stages,
            "inverse_depth": inverse,
            "top_inverse_depth_outliers_ranked_on": "D1_pre_anchor",
            "top_inverse_depth_outliers": self.top_records,
        }

    def summary_metrics(self) -> dict[str, float]:
        report = self.report()
        result: dict[str, float] = {}
        for stage, stats in report["stages"].items():
            for name in ("pred_min_m", "pred_mean_m", "pred_q0001_m", "pred_q001_m", "pred_q01_m", "pred_q1_m"):
                value = stats[name]
                result[f"audit_{stage}_{name}"] = float(value) if value is not None else float("nan")
            result[f"audit_{stage}_nonpositive_or_nonfinite"] = float(stats["nonpositive_or_nonfinite"])
            for name, count in stats["low_depth_counts"].items():
                result[f"audit_{stage}_{name}"] = float(count)
        for stage, stats in report["inverse_depth"].items():
            result[f"audit_{stage}_irmse_raw"] = float(stats["irmse_raw_1_per_km"])
            result[f"audit_{stage}_irmse_protocol"] = float(stats["irmse_protocol_1_per_km"])
            result[f"audit_{stage}_raw_invalid"] = float(stats["raw_invalid_pixels"])
        return result


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
