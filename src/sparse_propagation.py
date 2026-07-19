from __future__ import annotations

import torch
import torch.nn.functional as F


def downsample_depth_with_mask(
    depth: torch.Tensor,
    mask: torch.Tensor,
    scale: int = 4,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Valid-average downsample depth and mask.

    Args:
      depth: [B,1,H,W]
      mask: [B,1,H,W]

    Returns:
      depth_ds: [B,1,H/scale,W/scale]
      mask_ds: [B,1,H/scale,W/scale]
    """
    k = int(scale)
    mask_f = mask.float()
    numerator = F.avg_pool2d(depth.float() * mask_f, kernel_size=k, stride=k)
    denominator = F.avg_pool2d(mask_f, kernel_size=k, stride=k)
    depth_ds = numerator / denominator.clamp_min(eps)
    mask_ds = (denominator > 0).float()
    return depth_ds, mask_ds


def local_sparse_propagation(
    sparse: torch.Tensor,
    mask: torch.Tensor,
    scale: int = 4,
    kernel_schedule: tuple[int, ...] = (3, 5, 9, 17),
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compile-friendly sparse depth fill at coarse resolution.

    The original analytic KNN path is useful for ablations, but its global
    `cdist + topk` work is expensive on edge GPUs. This path downsamples sparse
    LiDAR to the model's coarse scale, then expands known depths with a few
    normalized-convolution passes. Remaining holes receive the per-sample mean
    sparse depth, keeping the output finite without a Python data-dependent
    branch.
    """
    depth_ds, valid_ds = downsample_depth_with_mask(sparse, mask, scale=scale, eps=eps)
    filled = depth_ds
    valid = valid_ds

    for kernel in kernel_schedule:
        k = int(kernel)
        pad = k // 2
        numerator = F.avg_pool2d(filled * valid, kernel_size=k, stride=1, padding=pad)
        denominator = F.avg_pool2d(valid, kernel_size=k, stride=1, padding=pad)
        proposal = numerator / denominator.clamp_min(eps)
        update = (valid <= 0.0) & (denominator > eps)
        filled = torch.where(update, proposal, filled)
        valid = torch.where(update, torch.ones_like(valid), valid)

    global_sum = (depth_ds * valid_ds).flatten(2).sum(dim=-1).view(depth_ds.shape[0], 1, 1, 1)
    global_count = valid_ds.flatten(2).sum(dim=-1).view(depth_ds.shape[0], 1, 1, 1)
    global_mean = global_sum / global_count.clamp_min(eps)
    return torch.where(valid > 0.0, filled, global_mean)


def knn_sparse_propagation(
    rgb: torch.Tensor,
    sparse: torch.Tensor,
    mask: torch.Tensor,
    ray: torch.Tensor,
    scale: int = 4,
    k: int = 4,
    alpha: float = 0.6,
    beta: float = 2.0,
    gamma: float = 0.5,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Analytic K-nearest bilateral propagation used only for ablations.

    This path uses global ``cdist + topk`` in chunks and is substantially more
    expensive than :func:`local_sparse_propagation` on deployment hardware.
    """
    if rgb.ndim != 4 or sparse.ndim != 4 or mask.ndim != 4 or ray.ndim != 4:
        raise ValueError("Expected BCHW tensors for rgb/sparse/mask/ray")
    B, _, H, W = rgb.shape
    Hs, Ws = H // scale, W // scale
    device = rgb.device
    dtype = rgb.dtype
    rgb_ds = F.interpolate(rgb, size=(Hs, Ws), mode="bilinear", align_corners=False)
    ray_ds = F.interpolate(ray, size=(Hs, Ws), mode="bilinear", align_corners=False)
    ray_ds = F.normalize(ray_ds, dim=1, eps=1e-6)
    out = torch.zeros((B, 1, Hs, Ws), device=device, dtype=dtype)

    ty, tx = torch.meshgrid(
        torch.arange(Hs, device=device, dtype=dtype),
        torch.arange(Ws, device=device, dtype=dtype),
        indexing="ij",
    )
    target_coord = torch.stack([ty.reshape(-1), tx.reshape(-1)], dim=1)  # [N,2]
    target_rgb = rgb_ds.permute(0, 2, 3, 1).reshape(B, Hs * Ws, 3)
    target_ray = ray_ds.permute(0, 2, 3, 1).reshape(B, Hs * Ws, 3)

    for b in range(B):
        valid_y, valid_x = torch.nonzero(mask[b, 0] > 0.5, as_tuple=True)
        if valid_y.numel() == 0:
            continue
        valid_depth = sparse[b, 0, valid_y, valid_x].to(dtype=dtype)  # [P]
        valid_coord = torch.stack([valid_y.to(dtype) / scale, valid_x.to(dtype) / scale], dim=1)  # [P,2]
        valid_rgb = rgb[b, :, valid_y, valid_x].transpose(0, 1).to(dtype=dtype)  # [P,3]
        valid_ray = ray[b, :, valid_y, valid_x].transpose(0, 1).to(dtype=dtype)  # [P,3]
        kk = min(int(k), int(valid_depth.numel()))
        flat = torch.empty((Hs * Ws,), device=device, dtype=dtype)

        for start in range(0, Hs * Ws, chunk_size):
            end = min(start + chunk_size, Hs * Ws)
            coord_chunk = target_coord[start:end]
            spatial = torch.cdist(coord_chunk, valid_coord, p=2.0)  # [chunk,P]
            dist, nn_idx = torch.topk(spatial, k=kk, dim=1, largest=False)
            q_rgb = valid_rgb[nn_idx]  # [chunk,k,3]
            q_ray = valid_ray[nn_idx]  # [chunk,k,3]
            color = (target_rgb[b, start:end, None, :] - q_rgb).abs().sum(dim=-1)
            ray_delta = (target_ray[b, start:end, None, :] - q_ray).abs().sum(dim=-1)
            logits = -alpha * dist - beta * color - gamma * ray_delta
            weights = torch.softmax(logits, dim=1)
            flat[start:end] = (weights * valid_depth[nn_idx]).sum(dim=1)

        out[b, 0] = flat.reshape(Hs, Ws)

    return out


def fast_sparse_propagation(
    rgb: torch.Tensor,
    sparse: torch.Tensor,
    mask: torch.Tensor,
    ray: torch.Tensor,
    scale: int = 4,
    k: int = 4,
    alpha: float = 0.6,
    beta: float = 2.0,
    gamma: float = 0.5,
    chunk_size: int = 4096,
    mode: str = "local",
) -> torch.Tensor:
    """Dispatch sparse propagation implementation.

    Modes:
      local: edge-friendly normalized-convolution fill.
      analytic/knn: original global KNN bilateral propagation.
    """
    mode_key = str(mode).lower()
    if mode_key in {"local", "edge", "normalized_conv", "normalized-conv"}:
        return local_sparse_propagation(sparse=sparse, mask=mask, scale=scale)
    if mode_key in {"analytic", "knn", "analytic_knn", "analytic-knn"}:
        return knn_sparse_propagation(
            rgb=rgb,
            sparse=sparse,
            mask=mask,
            ray=ray,
            scale=scale,
            k=k,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            chunk_size=chunk_size,
        )
    raise ValueError(f"Unsupported sparse propagation mode: {mode}")
