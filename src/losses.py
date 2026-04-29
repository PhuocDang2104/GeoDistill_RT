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


def scale_shift_invariant_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
    min_valid_pixels: int = 128,
    min_depth: float = 1e-3,
) -> torch.Tensor:
    """Scale-and-shift invariant dense structure loss.

    Shapes:
      pred: [B,1,h,w] metric student depth.
      target: [B,1,h,w] relative monocular depth, e.g. D_da_raw.
      mask: optional [B,1,h,w] validity mask.

    The fit is target-to-prediction:
      pred ~= a * target + b

    This keeps metric scale governed by sparse/GT/DMD3C supervision while
    Depth Anything contributes only relative structure.
    """
    if pred.ndim == 3:
        pred = pred[:, None]
    if target.ndim == 3:
        target = target[:, None]
    pred_f = pred.float()
    target_f = target.float()
    if mask is None:
        mask_b = torch.ones_like(pred_f, dtype=torch.bool)
    else:
        mask_b = mask.bool()
        if mask_b.ndim == 3:
            mask_b = mask_b[:, None]

    losses: list[torch.Tensor] = []
    for idx in range(pred_f.shape[0]):
        p = pred_f[idx, 0]
        t = target_f[idx, 0]
        valid = mask_b[idx, 0] & torch.isfinite(p.detach()) & torch.isfinite(t) & (p.detach() > min_depth)
        if int(valid.sum().item()) < int(min_valid_pixels):
            continue

        x = t[valid].detach()
        y = p[valid].detach()
        n = x.numel()
        sum_x = x.sum()
        sum_y = y.sum()
        sum_xx = (x * x).sum()
        sum_xy = (x * y).sum()
        denom = n * sum_xx - sum_x * sum_x
        if (not bool(torch.isfinite(denom).item())) or float(denom.abs().item()) <= eps:
            continue

        a = (n * sum_xy - sum_x * sum_y) / denom
        b = (sum_y - a * sum_x) / n
        aligned = a.detach() * t[valid] + b.detach()
        losses.append(F.smooth_l1_loss(p[valid], aligned, reduction="mean"))

    if not losses:
        return pred.new_tensor(0.0)
    return torch.stack(losses).mean().to(dtype=pred.dtype)


def geort_loss(
    pred: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    loss_cfg: dict[str, Any],
    schedule_cfg: dict[str, Any],
    epoch: int,
    mono_ssi_cfg: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the GeoRT objective without a normal loss.

    Prediction shapes:
      D_c/log_var: [B,1,H/4,W/4]

    Batch supervision shapes before downsampling:
      gt/sparse/mask/gt_mask: [B,1,H,W]
      D_teacher/C_teacher: [B,1,H/4,W/4] if available
      D_da_raw/da_raw_valid: [B,1,H,W] if mono SSI is enabled.
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

    L_ssi = D_c.new_tensor(0.0)
    if mono_ssi_cfg and bool(mono_ssi_cfg.get("enabled", False)) and epoch >= int(mono_ssi_cfg.get("start_epoch", 5)):
        mono_key = str(mono_ssi_cfg.get("key", "D_da_raw"))
        if mono_key in batch:
            D_da_raw = match_prediction_size(batch[mono_key].float(), target_hw, mode="bilinear")
            da_mask = batch.get("da_raw_valid")
            da_mask_ds = match_prediction_size(da_mask.float(), target_hw, mode="nearest") if da_mask is not None else None
            L_ssi = scale_shift_invariant_loss(
                D_c,
                D_da_raw,
                mask=da_mask_ds,
                min_valid_pixels=int(mono_ssi_cfg.get("min_valid_pixels", 128)),
                min_depth=float(mono_ssi_cfg.get("min_depth", 1e-3)),
            )

    total = (
        float(loss_cfg.get("lambda_gt", 1.0)) * L_gt
        + float(loss_cfg.get("lambda_T", 0.5)) * L_T
        + float(loss_cfg.get("lambda_S", 1.0)) * L_S
        + float(loss_cfg.get("lambda_C", 0.05)) * L_C
        + float(loss_cfg.get("lambda_E", 0.01)) * L_E
        + float((mono_ssi_cfg or {}).get("weight", 0.0)) * L_ssi
    )
    items = {
        "loss": float(total.detach().cpu()),
        "L_gt": float(L_gt.detach().cpu()),
        "L_T": float(L_T.detach().cpu()),
        "L_S": float(L_S.detach().cpu()),
        "L_C": float(L_C.detach().cpu()),
        "L_E": float(L_E.detach().cpu()),
        "L_ssi": float(L_ssi.detach().cpu()),
    }
    return total, items
