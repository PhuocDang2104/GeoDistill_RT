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
    w_dmd3c: np.ndarray
    D_full: np.ndarray
    C_full: np.ndarray


def robust_normalize_structure(x: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
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


def build_geometry_teacher(
    shape_hw: tuple[int, int],
    D_da_raw: np.ndarray | None = None,
    D_da_aligned: np.ndarray | None = None,
    D_m3d: np.ndarray | None = None,
    D_dmd3c: np.ndarray | None = None,
    min_depth: float = 1e-3,
    max_depth: float = 120.0,
    prior_da: float = 1.0,
    prior_m3d: float = 0.5,
    prior_dmd3c: float = 0.25,
) -> dict[str, np.ndarray]:
    """Build a separated structure teacher R_G/C_G.

    Relative Depth Anything raw is preferred for geometry. Metric teachers are
    converted through log-depth and robust normalization so they supervise only
    structure, not metric scale.
    """
    candidates: list[tuple[str, np.ndarray, np.ndarray, float]] = []

    if D_da_raw is not None:
        raw = _resize_depth_like(D_da_raw, shape_hw)
        valid = np.isfinite(raw)
        candidates.append(("da", robust_normalize_structure(raw, valid), valid, float(prior_da)))
    elif D_da_aligned is not None:
        da = _resize_depth_like(D_da_aligned, shape_hw)
        valid = np.isfinite(da) & (da > min_depth) & (da < max_depth)
        candidates.append(("da", robust_normalize_structure(np.log(np.clip(da, min_depth, max_depth)), valid), valid, float(prior_da)))

    if D_m3d is not None:
        m3d = _resize_depth_like(D_m3d, shape_hw)
        valid = np.isfinite(m3d) & (m3d > min_depth) & (m3d < max_depth)
        candidates.append(("m3d", robust_normalize_structure(np.log(np.clip(m3d, min_depth, max_depth)), valid), valid, float(prior_m3d)))

    if D_dmd3c is not None:
        dmd = _resize_depth_like(D_dmd3c, shape_hw)
        valid = np.isfinite(dmd) & (dmd > min_depth) & (dmd < max_depth)
        candidates.append(("dmd3c", robust_normalize_structure(np.log(np.clip(dmd, min_depth, max_depth)), valid), valid, float(prior_dmd3c)))

    h, w = shape_hw
    if not candidates:
        zeros = np.zeros((h, w), dtype=np.float32)
        return {"R_G": zeros, "C_G": zeros, "w_da": zeros, "w_m3d": zeros, "w_dmd3c": zeros}

    scores = []
    maps = []
    names = []
    for name, R, valid, prior in candidates:
        score = (float(prior) * valid.astype(np.float32)).astype(np.float32)
        scores.append(score)
        maps.append(R.astype(np.float32))
        names.append(name)
    score_stack = np.stack(scores, axis=0)
    denom = score_stack.sum(axis=0, keepdims=True)
    valid_any = denom[0] > 1e-8
    weights = np.zeros_like(score_stack, dtype=np.float32)
    weights[:, valid_any] = score_stack[:, valid_any] / denom[:, valid_any]

    R_G = np.zeros((h, w), dtype=np.float32)
    for idx, R in enumerate(maps):
        R_G += weights[idx] * R
    C_G = np.zeros((h, w), dtype=np.float32)
    C_G[valid_any] = np.max(weights[:, valid_any], axis=0)

    out = {"R_G": R_G.astype(np.float32), "C_G": C_G.astype(np.float32)}
    for key in ("da", "m3d", "dmd3c"):
        if key in names:
            out[f"w_{key}"] = weights[names.index(key)].astype(np.float32)
        else:
            out[f"w_{key}"] = np.zeros((h, w), dtype=np.float32)
    return out


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
    prior_m3d: float = 1.0,
    prior_da: float = 0.1,
    prior_dmd3c: float = 2.0,
) -> FusionResult:
    """DMD3C-dominant conflict-aware teacher fusion.

    Diagnostic weights still implement:
      q_i = exp(-alpha * (1 - <N(D_i), N_dsine>))
      r_i = exp(-beta * M * |D_i - S|)

    with teacher priors. The metric pseudo target is DMD3C wherever DMD3C
    is valid; weighted fusion is only a fallback for invalid DMD3C pixels.
    """
    shape_hw = D_m3d.shape
    D_m3d = _resize_depth_like(D_m3d, shape_hw)
    D_da_aligned = _resize_depth_like(D_da_aligned, shape_hw)
    sparse = _resize_depth_like(sparse, shape_hw)
    mask = _resize_depth_like(mask, shape_hw) > 0.5
    N_dsine = _resize_normals_like(N_dsine, shape_hw)

    teacher_names = ["m3d", "da"]
    depths = [D_m3d.astype(np.float32), D_da_aligned.astype(np.float32)]
    priors = [float(prior_m3d), float(prior_da)]
    D_dmd3c_resized = None
    if D_dmd3c is not None:
        teacher_names.append("dmd3c")
        D_dmd3c_resized = _resize_depth_like(D_dmd3c, shape_hw).astype(np.float32)
        depths.append(D_dmd3c_resized)
        priors.append(float(prior_dmd3c))
    scores = []
    for D, prior in zip(depths, priors):
        valid = np.isfinite(D) & (D > min_depth) & (D < max_depth)
        N_depth = depth_to_normals(D, K, min_depth=min_depth)
        dot = np.sum(N_depth * N_dsine, axis=0)
        valid_dot = valid & np.isfinite(dot)
        if valid_dot.any() and float(np.nanmean(dot[valid_dot])) < 0.0:
            N_depth = -N_depth
            dot = -dot
        dot = np.clip(dot, -1.0, 1.0)
        delta_normal = 1.0 - dot
        delta_sparse = np.where(mask, np.abs(D - sparse), 0.0)
        q = np.exp(-float(alpha_normal) * np.clip(delta_normal, 0.0, 2.0))
        r = np.exp(-float(beta_sparse) * np.clip(delta_sparse, 0.0, max_depth))
        score = (float(prior) * q * r).astype(np.float32)
        score[~valid] = 0.0
        score[~np.isfinite(score)] = 0.0
        scores.append(score)

    score_stack = np.stack(scores, axis=0)
    denom = np.sum(score_stack, axis=0, keepdims=True)
    valid_any = denom[0] > 1e-8
    weights = np.zeros_like(score_stack, dtype=np.float32)
    weights[:, valid_any] = score_stack[:, valid_any] / denom[:, valid_any]

    D_weighted = np.zeros(shape_hw, dtype=np.float32)
    for idx, D in enumerate(depths):
        D_weighted += weights[idx] * D
    D_weighted[~valid_any] = 0.0
    D_weighted = np.clip(D_weighted, 0.0, max_depth).astype(np.float32)

    if confidence_mode == "ones":
        C_full = valid_any.astype(np.float32)
    elif confidence_mode == "max_weight":
        C_full = np.max(weights, axis=0).astype(np.float32)
    else:
        raise ValueError(f"Unsupported confidence_mode: {confidence_mode}")

    D_full = D_weighted.copy()
    if D_dmd3c_resized is not None:
        dmd_valid = np.isfinite(D_dmd3c_resized) & (D_dmd3c_resized > min_depth) & (D_dmd3c_resized < max_depth)
        D_full[dmd_valid] = D_dmd3c_resized[dmd_valid]
        C_full[dmd_valid] = 1.0
    D_full = np.clip(D_full, 0.0, max_depth).astype(np.float32)
    C_full = np.clip(C_full, 0.0, 1.0).astype(np.float32)

    D_teacher = downsample_map(D_full, output_scale, interpolation=cv2.INTER_AREA)
    C_teacher = downsample_map(C_full, output_scale, interpolation=cv2.INTER_AREA)
    w_m3d = downsample_map(weights[0], output_scale, interpolation=cv2.INTER_AREA)
    w_da = downsample_map(weights[1], output_scale, interpolation=cv2.INTER_AREA)
    if "dmd3c" in teacher_names:
        w_dmd3c = downsample_map(weights[2], output_scale, interpolation=cv2.INTER_AREA)
    else:
        w_dmd3c = np.zeros_like(w_m3d, dtype=np.float32)
    return FusionResult(
        D_teacher=D_teacher.astype(np.float32),
        C_teacher=C_teacher.astype(np.float32),
        w_m3d=w_m3d.astype(np.float32),
        w_da=w_da.astype(np.float32),
        w_dmd3c=w_dmd3c.astype(np.float32),
        D_full=D_full,
        C_full=C_full,
    )
