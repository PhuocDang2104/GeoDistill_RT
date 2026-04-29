from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FusionResult:
    D_teacher: np.ndarray
    C_teacher: np.ndarray
    w_m3d: np.ndarray
    w_da: np.ndarray
    w_dmd3c: np.ndarray | None
    D_full: np.ndarray
    C_full: np.ndarray


def _resize_depth_like(depth: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    if depth.shape == shape_hw:
        return depth.astype(np.float32)
    h, w = shape_hw
    return cv2.resize(depth.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def _resize_normals_like(normals: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    if normals.ndim == 3 and normals.shape[0] == 3:
        chw = normals
    elif normals.ndim == 3 and normals.shape[-1] == 3:
        chw = normals.transpose(2, 0, 1)
    else:
        raise ValueError(f"Expected normals [3,H,W] or [H,W,3], got {normals.shape}")
    h, w = shape_hw
    if chw.shape[1:] != shape_hw:
        resized = [cv2.resize(chw[c].astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) for c in range(3)]
        chw = np.stack(resized, axis=0)
    norm = np.linalg.norm(chw, axis=0, keepdims=True).clip(min=1e-6)
    return (chw / norm).astype(np.float32)


def depth_to_normals(depth: np.ndarray, K: np.ndarray, min_depth: float = 1e-3) -> np.ndarray:
    """Derive camera-space surface normals from metric z-depth.

    Args:
      depth: float [H,W], metric z-depth.
      K: float [3,3] camera intrinsics.

    Returns:
      normals: float32 [3,H,W], unit length. Invalid depth gets zero normal.
    """
    depth = depth.astype(np.float32)
    h, w = depth.shape
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    u, v = np.meshgrid(xs, ys)
    points = np.stack([(u - cx) / fx * depth, (v - cy) / fy * depth, depth], axis=-1)

    dPdy = np.gradient(points, axis=0)
    dPdx = np.gradient(points, axis=1)
    normals = np.cross(dPdx, dPdy)
    normals = normals.transpose(2, 0, 1)
    norm = np.linalg.norm(normals, axis=0, keepdims=True).clip(min=1e-6)
    normals = normals / norm
    valid = np.isfinite(depth) & (depth > min_depth)
    normals[:, ~valid] = 0.0
    return normals.astype(np.float32)


def downsample_map(array: np.ndarray, scale: int = 4, interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    h, w = array.shape[-2:]
    out_h = max(1, h // scale)
    out_w = max(1, w // scale)
    if array.ndim == 2:
        return cv2.resize(array.astype(np.float32), (out_w, out_h), interpolation=interpolation).astype(np.float32)
    if array.ndim == 3:
        channels = [cv2.resize(array[c].astype(np.float32), (out_w, out_h), interpolation=interpolation) for c in range(array.shape[0])]
        return np.stack(channels, axis=0).astype(np.float32)
    raise ValueError(f"Unsupported array shape for downsample: {array.shape}")


def fuse_teachers(
    D_m3d: np.ndarray,
    D_da_aligned: np.ndarray,
    N_dsine: np.ndarray,
    sparse: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    D_dmd3c: np.ndarray | None = None,
    alpha_normal: float = 1.0,
    beta_sparse: float = 1.0,
    output_scale: int = 4,
    confidence_mode: str = "max_weight",
    min_depth: float = 1e-3,
    max_depth: float = 120.0,
) -> FusionResult:
    """Conflict-aware teacher fusion.

    Implements:
      q_i = exp(-alpha * (1 - <N(D_i), N_dsine>))
      r_i = exp(-beta * M * |D_i - S|)
      w_i = q_i r_i / sum_j q_j r_j
      D_T = sum_i w_i D_i
      C_T = max_i w_i
    """
    shape_hw = D_m3d.shape
    D_m3d = _resize_depth_like(D_m3d, shape_hw)
    D_da_aligned = _resize_depth_like(D_da_aligned, shape_hw)
    sparse = _resize_depth_like(sparse, shape_hw)
    mask = _resize_depth_like(mask, shape_hw) > 0.5
    N_dsine = _resize_normals_like(N_dsine, shape_hw)

    teacher_names = ["m3d", "da"]
    depths = [D_m3d.astype(np.float32), D_da_aligned.astype(np.float32)]
    if D_dmd3c is not None:
        teacher_names.append("dmd3c")
        depths.append(_resize_depth_like(D_dmd3c, shape_hw).astype(np.float32))
    scores = []
    normals = []
    for D in depths:
        valid = np.isfinite(D) & (D > min_depth) & (D < max_depth)
        N_depth = depth_to_normals(D, K, min_depth=min_depth)
        dot = np.sum(N_depth * N_dsine, axis=0)
        valid_dot = valid & np.isfinite(dot)
        if valid_dot.any() and float(np.nanmean(dot[valid_dot])) < 0.0:
            N_depth = -N_depth
            dot = -dot
        normals.append(N_depth)
        dot = np.clip(dot, -1.0, 1.0)
        delta_normal = 1.0 - dot
        delta_sparse = np.where(mask, np.abs(D - sparse), 0.0)
        q = np.exp(-float(alpha_normal) * np.clip(delta_normal, 0.0, 2.0))
        r = np.exp(-float(beta_sparse) * np.clip(delta_sparse, 0.0, max_depth))
        score = (q * r).astype(np.float32)
        score[~valid] = 0.0
        score[~np.isfinite(score)] = 0.0
        scores.append(score)

    score_stack = np.stack(scores, axis=0)
    denom = np.sum(score_stack, axis=0, keepdims=True)
    valid_any = denom[0] > 1e-8
    weights = np.zeros_like(score_stack, dtype=np.float32)
    weights[:, valid_any] = score_stack[:, valid_any] / denom[:, valid_any]

    D_full = np.zeros(shape_hw, dtype=np.float32)
    for idx, D in enumerate(depths):
        D_full += weights[idx] * D
    D_full[~valid_any] = 0.0
    D_full = np.clip(D_full, 0.0, max_depth).astype(np.float32)

    if confidence_mode == "ones":
        C_full = valid_any.astype(np.float32)
    elif confidence_mode == "max_weight":
        C_full = np.max(weights, axis=0).astype(np.float32)
    else:
        raise ValueError(f"Unsupported confidence_mode: {confidence_mode}")

    D_teacher = downsample_map(D_full, output_scale, interpolation=cv2.INTER_AREA)
    C_teacher = downsample_map(C_full, output_scale, interpolation=cv2.INTER_AREA)
    w_m3d = downsample_map(weights[0], output_scale, interpolation=cv2.INTER_AREA)
    w_da = downsample_map(weights[1], output_scale, interpolation=cv2.INTER_AREA)
    w_dmd3c = downsample_map(weights[2], output_scale, interpolation=cv2.INTER_AREA) if "dmd3c" in teacher_names else None
    return FusionResult(
        D_teacher=D_teacher.astype(np.float32),
        C_teacher=C_teacher.astype(np.float32),
        w_m3d=w_m3d.astype(np.float32),
        w_da=w_da.astype(np.float32),
        w_dmd3c=w_dmd3c.astype(np.float32) if w_dmd3c is not None else None,
        D_full=D_full,
        C_full=C_full,
    )
