from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .sparse_propagation import downsample_depth_with_mask


def huber(residual: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    abs_r = residual.abs()
    quad = torch.minimum(abs_r, torch.tensor(delta, dtype=residual.dtype, device=residual.device))
    lin = abs_r - quad
    return 0.5 * quad * quad + delta * lin


def mean_valid(value: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mask_f = mask.to(dtype=value.dtype)
    return (value * mask_f).sum() / mask_f.sum().clamp_min(eps)


def match_prediction_size(x: torch.Tensor, target_hw: tuple[int, int], mode: str = "nearest") -> torch.Tensor:
    if x.shape[-2:] == target_hw:
        return x
    if mode == "nearest":
        return F.interpolate(x, size=target_hw, mode=mode)
    return F.interpolate(x, size=target_hw, mode=mode, align_corners=False)


def edge_smoothness_loss(D_c: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    """Edge-aware smoothness.

    D_c: [B,1,h,w]
    rgb: [B,3,H,W], downsampled internally to [B,3,h,w]
    """
    rgb_down = F.interpolate(rgb, size=D_c.shape[-2:], mode="bilinear", align_corners=False)
    dx_d = (D_c[..., :, 1:] - D_c[..., :, :-1]).abs()
    dy_d = (D_c[..., 1:, :] - D_c[..., :-1, :]).abs()
    dx_i = (rgb_down[..., :, 1:] - rgb_down[..., :, :-1]).abs().mean(dim=1, keepdim=True)
    dy_i = (rgb_down[..., 1:, :] - rgb_down[..., :-1, :]).abs().mean(dim=1, keepdim=True)
    return (dx_d * torch.exp(-dx_i)).mean() + (dy_d * torch.exp(-dy_i)).mean()


def geort_loss(
    pred: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    loss_cfg: dict[str, Any],
    schedule_cfg: dict[str, Any],
    epoch: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the GeoRT objective without a normal loss.

    Prediction shapes:
      D_c/log_var: [B,1,H/4,W/4]

    Batch supervision shapes before downsampling:
      gt/sparse/mask/gt_mask: [B,1,H,W]
      D_teacher/C_teacher: [B,1,H/4,W/4] if available
    """
    D_c = pred["D_c"]
    log_var = pred["log_var"]
    target_hw = D_c.shape[-2:]
    full_h = batch["sparse"].shape[-2]
    scale = max(1, full_h // target_hw[0])

    gt_ds, gt_mask_ds = downsample_depth_with_mask(batch["gt"], batch["gt_mask"], scale=scale)
    sparse_ds, sparse_mask_ds = downsample_depth_with_mask(batch["sparse"], batch["mask"], scale=scale)
    gt_ds = match_prediction_size(gt_ds, target_hw)
    gt_mask_ds = match_prediction_size(gt_mask_ds, target_hw)
    sparse_ds = match_prediction_size(sparse_ds, target_hw)
    sparse_mask_ds = match_prediction_size(sparse_mask_ds, target_hw)

    if "D_teacher" in batch:
        D_teacher = match_prediction_size(batch["D_teacher"].float(), target_hw, mode="bilinear")
        C_teacher = match_prediction_size(batch.get("C_teacher", torch.ones_like(D_teacher)).float(), target_hw, mode="bilinear")
    else:
        D_teacher = torch.zeros_like(D_c)
        C_teacher = torch.zeros_like(D_c)
    teacher_mask = torch.isfinite(D_teacher) & (D_teacher > 1e-3)

    L_gt = mean_valid(huber(D_c - gt_ds), gt_mask_ds > 0.5)
    L_S = mean_valid((D_c - sparse_ds).abs(), sparse_mask_ds > 0.5)
    L_E = edge_smoothness_loss(D_c, batch["rgb"])

    add_teacher_epoch = int(schedule_cfg.get("add_teacher_epoch", 5))
    add_conf_epoch = int(schedule_cfg.get("add_confidence_epoch", 15))
    use_teacher_conf = bool(loss_cfg.get("use_teacher_conf", True)) and epoch >= add_conf_epoch

    if epoch >= add_teacher_epoch:
        teacher_weight = C_teacher.clamp(0.0, 1.0) if use_teacher_conf else torch.ones_like(C_teacher)
        L_T = mean_valid(teacher_weight * huber(D_c - D_teacher), teacher_mask)
    else:
        L_T = D_c.new_tensor(0.0)

    if epoch >= add_conf_epoch:
        gt_valid = gt_mask_ds > 0.5
        D_sup = torch.where(gt_valid, gt_ds, D_teacher)
        sup_mask = gt_valid | teacher_mask
        conf_term = torch.exp(-log_var) * huber(D_c - D_sup) + float(loss_cfg.get("lambda_s", 0.01)) * log_var
        L_C = mean_valid(conf_term, sup_mask)
    else:
        L_C = D_c.new_tensor(0.0)

    total = (
        float(loss_cfg.get("lambda_gt", 1.0)) * L_gt
        + float(loss_cfg.get("lambda_T", 0.5)) * L_T
        + float(loss_cfg.get("lambda_S", 1.0)) * L_S
        + float(loss_cfg.get("lambda_C", 0.05)) * L_C
        + float(loss_cfg.get("lambda_E", 0.01)) * L_E
    )
    items = {
        "loss": float(total.detach().cpu()),
        "L_gt": float(L_gt.detach().cpu()),
        "L_T": float(L_T.detach().cpu()),
        "L_S": float(L_S.detach().cpu()),
        "L_C": float(L_C.detach().cpu()),
        "L_E": float(L_E.detach().cpu()),
    }
    return total, items
