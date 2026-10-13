"""GT-priority metric distillation; every reduction remains on the accelerator."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def range_rmse(pred, gt, valid, min_pixels=64):
    """A small auxiliary term aligned with far-range squared-error evaluation.

    Tensorized bins; the very rare >80 m bin has 1/4 of a regular bin's vote.
    Insufficient-support bins do not become noisy single-pixel objectives.
    """
    lower = pred.new_tensor([0, 20, 40, 60, 80]).view(1, 5, 1, 1)
    upper = pred.new_tensor([20, 40, 60, 80, 120]).view(1, 5, 1, 1)
    masks = ((gt >= lower) & (gt < upper) & valid.bool()).float()
    counts = masks.sum((0, 2, 3))
    mse = ((pred - gt).square() * masks).sum((0, 2, 3)) / counts.clamp_min(1)
    rmse = (mse + 1e-6).sqrt() - 1e-3
    weights = pred.new_tensor([1, 1, 1, 1, .25]) * (counts >= min_pixels)
    return (weights * rmse).sum() / weights.sum().clamp_min(1), (weights > 0).sum()


def pooled(value, support, shape):
    factor = value.shape[-1] // shape[-1]
    if factor == 1:
        return value, support
    weight = F.avg_pool2d(support, factor, factor)
    target = F.avg_pool2d(value * support, factor, factor) / weight.clamp_min(1e-6)
    return target, weight


def mean_masked(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1)


def huber(error, delta=1.0):
    absolute = error.abs()
    return torch.where(absolute <= delta, 0.5 * error.square(), delta * (absolute - 0.5 * delta))


def pair_gradient(pred, target, weight):
    px, py = pred[..., 1:] - pred[..., :-1], pred[..., 1:, :] - pred[..., :-1, :]
    tx, ty = target[..., 1:] - target[..., :-1], target[..., 1:, :] - target[..., :-1, :]
    wx = torch.minimum(weight[..., 1:], weight[..., :-1])
    wy = torch.minimum(weight[..., 1:, :], weight[..., :-1, :])
    # Confidence is not normalized away: all-low-confidence maps remain weak.
    x = (huber(px - tx, 0.05) * wx).sum() / (wx > 0).sum().clamp_min(1)
    y = (huber(py - ty, 0.05) * wy).sum() / (wy > 0).sum().clamp_min(1)
    return x + y


def teacher_weights(batch, threshold):
    teacher, confidence = batch["teacher"].float(), batch["confidence"].float()
    valid = torch.isfinite(teacher) & (teacher > 0.1) & (teacher < 120)
    valid = valid & torch.isfinite(confidence) & (confidence >= threshold)
    forbidden = (batch["gt_mask"] > 0) | (batch["mask"] > 0)
    eligible = valid & ~forbidden
    teacher = torch.where(valid, teacher, torch.ones_like(teacher))
    confidence = torch.where(eligible, confidence.clamp(0, 1), torch.zeros_like(confidence))
    return teacher, confidence, forbidden.float(), eligible
