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
    C_dmd3c: np.ndarray


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


def _resize_rgb_like(rgb: np.ndarray | None, shape_hw: tuple[int, int]) -> np.ndarray | None:
    if rgb is None:
        return None
    arr = rgb.astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = arr.transpose(1, 2, 0)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return None
    h, w = shape_hw
    if arr.shape[:2] != shape_hw:
        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
    if float(np.nanmax(arr)) > 2.0:
        arr = arr / 255.0
    arr[~np.isfinite(arr)] = 0.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _normalize_positive(x: np.ndarray, valid: np.ndarray | None = None, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    valid_mask = np.isfinite(arr) if valid is None else (valid & np.isfinite(arr))
    if int(valid_mask.sum()) < 32:
        out = np.zeros_like(arr, dtype=np.float32)
        out[np.isfinite(arr)] = np.clip(arr[np.isfinite(arr)], 0.0, None)
        scale = float(np.nanmax(out)) if np.isfinite(out).any() else 0.0
    else:
        values = np.clip(arr[valid_mask], 0.0, None)
        scale = float(np.percentile(values, 95.0))
    if scale <= eps or not np.isfinite(scale):
        return np.zeros_like(arr, dtype=np.float32)
    out = np.clip(arr, 0.0, None) / scale
    out[~np.isfinite(out)] = 0.0
    return np.clip(out, 0.0, 10.0).astype(np.float32)


def _gradient_magnitude(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    gy, gx = np.gradient(arr)
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def _image_edge(rgb: np.ndarray | None, shape_hw: tuple[int, int]) -> np.ndarray:
    img = _resize_rgb_like(rgb, shape_hw)
    if img is None:
        return np.zeros(shape_hw, dtype=np.float32)
    gray = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)
    return _normalize_positive(_gradient_magnitude(gray))


def _nearest_sparse_relative_error(
    depth: np.ndarray,
    sparse: np.ndarray,
    mask: np.ndarray,
    min_depth: float,
    max_depth: float,
    blend_radius: float,
) -> np.ndarray:
    valid_depth = np.isfinite(depth) & (depth > min_depth) & (depth < max_depth)
    anchor = (mask > 0.5) & np.isfinite(sparse) & (sparse > min_depth) & (sparse < max_depth) & valid_depth
    if int(anchor.sum()) < 1:
        return np.zeros_like(depth, dtype=np.float32)

    rel_error = np.abs(depth[anchor] - sparse[anchor]) / np.clip(sparse[anchor], min_depth, max_depth)
    try:
        src = np.ones(depth.shape, dtype=np.uint8)
        src[anchor] = 0
        dist, labels = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        max_label = int(labels.max())
        label_error = np.zeros(max_label + 1, dtype=np.float32)
        label_error.fill(float(np.median(rel_error)))
        anchor_labels = labels[anchor].reshape(-1)
        for lab, err in zip(anchor_labels, rel_error.reshape(-1)):
            lab_i = int(lab)
            if 0 <= lab_i <= max_label:
                label_error[lab_i] = min(label_error[lab_i], float(err))
        dense_error = label_error[labels]
        if blend_radius > 0:
            far_blend = 1.0 - np.exp(-dist.astype(np.float32) / float(blend_radius))
            dense_error = dense_error * (1.0 - far_blend)
        dense_error[~valid_depth] = 0.0
        return np.clip(dense_error, 0.0, 10.0).astype(np.float32)
    except Exception:
        out = np.zeros_like(depth, dtype=np.float32)
        out[anchor] = rel_error.astype(np.float32)
        return np.clip(out, 0.0, 10.0).astype(np.float32)


def _sparse_confidence(
    depth: np.ndarray,
    sparse: np.ndarray,
    mask: np.ndarray,
    min_depth: float,
    max_depth: float,
    decay: float,
    blend_radius: float,
) -> np.ndarray:
    rel_error = _nearest_sparse_relative_error(depth, sparse, mask, min_depth, max_depth, blend_radius)
    conf = np.exp(-float(decay) * rel_error).astype(np.float32)
    conf[~np.isfinite(conf)] = 0.0
    return np.clip(conf, 0.0, 1.0).astype(np.float32)


def _normal_confidence(
    depth: np.ndarray,
    N_dsine: np.ndarray | None,
    K: np.ndarray,
    rgb: np.ndarray | None,
    min_depth: float,
    max_depth: float,
    alpha: float,
) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > min_depth) & (depth < max_depth)
    if N_dsine is None:
        return valid.astype(np.float32)
    normals_ref = _resize_normals_like(N_dsine, depth.shape)
    N_depth = depth_to_normals(depth, K, min_depth=min_depth)
    dot = np.sum(N_depth * normals_ref, axis=0)
    valid_dot = valid & np.isfinite(dot)
    if valid_dot.any() and float(np.nanmean(dot[valid_dot])) < 0.0:
        dot = -dot
    delta = np.clip(1.0 - np.clip(dot, -1.0, 1.0), 0.0, 2.0)
    img_edge = _image_edge(rgb, depth.shape)
    depth_edge = _normalize_positive(_gradient_magnitude(np.clip(depth, min_depth, max_depth)), valid)
    plane_weight = np.exp(-img_edge).astype(np.float32) * np.exp(-depth_edge).astype(np.float32)
    conf = np.exp(-float(alpha) * plane_weight * delta).astype(np.float32)
    conf[~valid] = 0.0
    conf[~np.isfinite(conf)] = 0.0
    return np.clip(conf, 0.0, 1.0).astype(np.float32)


def _depth_edge_confidence(
    depth: np.ndarray,
    rgb: np.ndarray | None,
    min_depth: float,
    max_depth: float,
    decay: float,
) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > min_depth) & (depth < max_depth)
    img_edge = _image_edge(rgb, depth.shape)
    depth_edge = _normalize_positive(_gradient_magnitude(np.clip(depth, min_depth, max_depth)), valid)
    risk = img_edge * (0.5 + depth_edge)
    conf = np.exp(-float(decay) * risk).astype(np.float32)
    conf[~valid] = 0.0
    return np.clip(conf, 0.0, 1.0).astype(np.float32)


def _structure_edge_confidence(
    structure: np.ndarray,
    rgb: np.ndarray | None,
    decay: float,
    kappa: float = 4.0,
) -> np.ndarray:
    valid = np.isfinite(structure)
    r_edge = _normalize_positive(_gradient_magnitude(structure), valid)
    img_edge = _image_edge(rgb, structure.shape)
    unsupported = r_edge * np.exp(-float(kappa) * img_edge)
    conf = np.exp(-float(decay) * unsupported).astype(np.float32)
    conf[~valid] = 0.0
    return np.clip(conf, 0.0, 1.0).astype(np.float32)


def _geometry_reference_from_metric_teachers(
    shape_hw: tuple[int, int],
    D_da_aligned: np.ndarray | None,
    D_m3d: np.ndarray | None,
    min_depth: float,
    max_depth: float,
    prior_da: float,
    prior_m3d: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    candidates: list[tuple[np.ndarray, np.ndarray, float]] = []
    if D_da_aligned is not None:
        da = _resize_depth_like(D_da_aligned, shape_hw)
        valid = np.isfinite(da) & (da > min_depth) & (da < max_depth)
        if int(valid.sum()) >= 32:
            candidates.append((robust_normalize_structure(np.log(np.clip(da, min_depth, max_depth)), valid), valid, float(prior_da)))
    if D_m3d is not None:
        m3d = _resize_depth_like(D_m3d, shape_hw)
        valid = np.isfinite(m3d) & (m3d > min_depth) & (m3d < max_depth)
        if int(valid.sum()) >= 32:
            candidates.append((robust_normalize_structure(np.log(np.clip(m3d, min_depth, max_depth)), valid), valid, float(prior_m3d)))
    if not candidates:
        return None, None
    scores = np.stack([prior * valid.astype(np.float32) for _, valid, prior in candidates], axis=0)
    denom = scores.sum(axis=0, keepdims=True)
    valid_any = denom[0] > 1e-8
    weights = np.zeros_like(scores, dtype=np.float32)
    weights[:, valid_any] = scores[:, valid_any] / denom[:, valid_any]
    ref = np.zeros(shape_hw, dtype=np.float32)
    for idx, (R, _, _) in enumerate(candidates):
        ref += weights[idx] * R.astype(np.float32)
    return ref.astype(np.float32), valid_any


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
    N_dsine: np.ndarray | None = None,
    sparse: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    K: np.ndarray | None = None,
    rgb: np.ndarray | None = None,
    min_depth: float = 1e-3,
    max_depth: float = 120.0,
    prior_da: float = 1.0,
    prior_m3d: float = 0.5,
    prior_dmd3c: float = 0.25,
    alpha_normal: float = 1.0,
    beta_sparse: float = 1.0,
    edge_decay: float = 1.0,
    sparse_blend_radius: float = 48.0,
) -> dict[str, np.ndarray]:
    """Build a separated structure teacher R_G/C_G.

    Relative Depth Anything raw is preferred for geometry. Metric teachers are
    converted through log-depth and robust normalization so they supervise only
    structure, not metric scale.
    """
    candidates: list[tuple[str, np.ndarray, np.ndarray, float, np.ndarray | None]] = []

    if D_da_raw is not None:
        raw = _resize_depth_like(D_da_raw, shape_hw)
        valid = np.isfinite(raw)
        metric_like = None
        if D_da_aligned is not None:
            da_metric = _resize_depth_like(D_da_aligned, shape_hw)
            if int((np.isfinite(da_metric) & (da_metric > min_depth) & (da_metric < max_depth)).sum()) >= 32:
                metric_like = da_metric
        candidates.append(("da", robust_normalize_structure(raw, valid), valid, float(prior_da), metric_like))
    elif D_da_aligned is not None:
        da = _resize_depth_like(D_da_aligned, shape_hw)
        valid = np.isfinite(da) & (da > min_depth) & (da < max_depth)
        candidates.append(("da", robust_normalize_structure(np.log(np.clip(da, min_depth, max_depth)), valid), valid, float(prior_da), da))

    if D_m3d is not None:
        m3d = _resize_depth_like(D_m3d, shape_hw)
        valid = np.isfinite(m3d) & (m3d > min_depth) & (m3d < max_depth)
        candidates.append(("m3d", robust_normalize_structure(np.log(np.clip(m3d, min_depth, max_depth)), valid), valid, float(prior_m3d), m3d))

    if D_dmd3c is not None:
        dmd = _resize_depth_like(D_dmd3c, shape_hw)
        valid = np.isfinite(dmd) & (dmd > min_depth) & (dmd < max_depth)
        candidates.append(("dmd3c", robust_normalize_structure(np.log(np.clip(dmd, min_depth, max_depth)), valid), valid, float(prior_dmd3c), dmd))

    h, w = shape_hw
    if not candidates:
        zeros = np.zeros((h, w), dtype=np.float32)
        return {"R_G": zeros, "C_G": zeros, "w_da": zeros, "w_m3d": zeros, "w_dmd3c": zeros}

    scores = []
    maps = []
    names = []
    sparse_arr = _resize_depth_like(sparse, shape_hw) if sparse is not None else None
    mask_arr = (_resize_depth_like(mask, shape_hw) > 0.5) if mask is not None else None
    for name, R, valid, prior, metric_like in candidates:
        score = (float(prior) * valid.astype(np.float32)).astype(np.float32)
        score *= _structure_edge_confidence(R, rgb, decay=edge_decay)
        if metric_like is not None and K is not None:
            score *= _normal_confidence(metric_like, N_dsine, K, rgb, min_depth, max_depth, alpha_normal)
            if sparse_arr is not None and mask_arr is not None:
                score *= _sparse_confidence(metric_like, sparse_arr, mask_arr, min_depth, max_depth, beta_sparse, sparse_blend_radius)
        score[~valid] = 0.0
        score[~np.isfinite(score)] = 0.0
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
    rgb: np.ndarray | None = None,
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
    conf_min: float = 0.05,
    sparse_conf_decay: float = 6.0,
    sparse_blend_radius: float = 48.0,
    range_conf_decay: float = 0.25,
    edge_conf_decay: float = 1.0,
    geometry_conf_decay: float = 0.5,
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
    rgb = _resize_rgb_like(rgb, shape_hw)

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
        q = _normal_confidence(D, N_dsine, K, rgb, min_depth, max_depth, alpha_normal)
        r = _sparse_confidence(D, sparse, mask, min_depth, max_depth, beta_sparse, sparse_blend_radius)
        edge_conf = _depth_edge_confidence(D, rgb, min_depth, max_depth, edge_conf_decay)
        range_conf = np.exp(-float(range_conf_decay) * np.clip(D, 0.0, max_depth) / float(max_depth)).astype(np.float32)
        score = (float(prior) * q * r * edge_conf * range_conf).astype(np.float32)
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
    C_dmd3c = np.zeros(shape_hw, dtype=np.float32)
    if D_dmd3c_resized is not None:
        dmd_valid = np.isfinite(D_dmd3c_resized) & (D_dmd3c_resized > min_depth) & (D_dmd3c_resized < max_depth)
        c_sparse = _sparse_confidence(
            D_dmd3c_resized,
            sparse,
            mask,
            min_depth,
            max_depth,
            sparse_conf_decay,
            sparse_blend_radius,
        )
        geom_ref, geom_valid = _geometry_reference_from_metric_teachers(
            shape_hw,
            D_da_aligned,
            D_m3d,
            min_depth,
            max_depth,
            prior_da=max(prior_da, 1e-6),
            prior_m3d=max(prior_m3d, 1e-6),
        )
        if geom_ref is None or geom_valid is None:
            c_geom = np.ones(shape_hw, dtype=np.float32)
        else:
            dmd_structure = robust_normalize_structure(np.log(np.clip(D_dmd3c_resized, min_depth, max_depth)), dmd_valid)
            geom_error = np.abs(dmd_structure - geom_ref)
            c_geom = np.ones(shape_hw, dtype=np.float32)
            c_geom[geom_valid] = np.exp(-float(geometry_conf_decay) * np.clip(geom_error[geom_valid], 0.0, 10.0)).astype(np.float32)
        c_edge = _depth_edge_confidence(D_dmd3c_resized, rgb, min_depth, max_depth, edge_conf_decay)
        c_range = np.exp(-float(range_conf_decay) * np.clip(D_dmd3c_resized, 0.0, max_depth) / float(max_depth)).astype(np.float32)
        C_dmd3c = c_sparse * c_geom * c_edge * c_range
        C_dmd3c = np.clip(C_dmd3c, float(conf_min), 1.0).astype(np.float32)
        C_dmd3c[~dmd_valid] = 0.0
        D_full[dmd_valid] = D_dmd3c_resized[dmd_valid]
        C_full[dmd_valid] = C_dmd3c[dmd_valid]
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
        C_dmd3c=C_dmd3c.astype(np.float32),
    )
