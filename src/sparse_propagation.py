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
) -> torch.Tensor:
    """Analytic K-nearest sparse propagation.

    Args:
      rgb: [B,3,H,W] in [0,1]
      sparse: [B,1,H,W] metric sparse depth
      mask: [B,1,H,W] valid sparse mask
      ray: [B,3,H,W] normalized camera ray map

    Returns:
      D_init: [B,1,H/scale,W/scale] metric coarse dense prior.
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
