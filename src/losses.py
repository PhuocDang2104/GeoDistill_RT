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


def downsample_depth_with_conf(depth: torch.Tensor, conf: torch.Tensor, scale: int) -> tuple[torch.Tensor, torch.Tensor]:
    weight = ((depth > 1e-3) & torch.isfinite(depth) & (conf > 0)).float() * conf.float().clamp(0.0, 1.0)
    numerator = F.avg_pool2d(depth.float() * weight, kernel_size=scale, stride=scale)
    denominator = F.avg_pool2d(weight, kernel_size=scale, stride=scale)
    depth_ds = numerator / denominator.clamp_min(1e-6)
    conf_ds = F.avg_pool2d(weight, kernel_size=scale, stride=scale).clamp(0.0, 1.0)
    return depth_ds, conf_ds


def edge_smoothness_full(D_full: torch.Tensor, rgb: torch.Tensor, max_depth: float = 120.0) -> torch.Tensor:
    depth = D_full / float(max_depth)
    dx_d = (depth[..., :, 1:] - depth[..., :, :-1]).abs()
    dy_d = (depth[..., 1:, :] - depth[..., :-1, :]).abs()
    dx_i = (rgb[..., :, 1:] - rgb[..., :, :-1]).abs().mean(dim=1, keepdim=True)
    dy_i = (rgb[..., 1:, :] - rgb[..., :-1, :]).abs().mean(dim=1, keepdim=True)
    return (dx_d * torch.exp(-dx_i)).mean() + (dy_d * torch.exp(-dy_i)).mean()


def geometry_ssi_loss(
    D_full: torch.Tensor,
    R_G: torch.Tensor,
    C_G: torch.Tensor,
    conf_threshold: float = 0.05,
    min_valid_pixels: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit R_G -> log(D_full) and regress log-depth to the fitted structure."""
    log_d = torch.log(D_full.clamp_min(1e-3))
    losses: list[torch.Tensor] = []
    alphas: list[torch.Tensor] = []
    for idx in range(log_d.shape[0]):
        pred = log_d[idx, 0]
        target = R_G[idx, 0]
        conf = C_G[idx, 0].clamp(0.0, 1.0)
        valid = (conf > conf_threshold) & torch.isfinite(pred.detach()) & torch.isfinite(target)
        if int(valid.sum().item()) < min_valid_pixels:
            alphas.append(pred.new_tensor(1.0))
            continue

        x = target[valid].detach().float()
        y = pred[valid].detach().float()
        w = conf[valid].detach().float().clamp_min(1e-4)
        sw = w.sum().clamp_min(1e-6)
        mx = (w * x).sum() / sw
        my = (w * y).sum() / sw
        var = (w * (x - mx) * (x - mx)).sum()
        if float(var.detach().abs().cpu()) < 1e-8:
            alphas.append(pred.new_tensor(1.0))
            continue

        cov = (w * (x - mx) * (y - my)).sum()
        alpha = cov / var.clamp_min(1e-8)
        beta = my - alpha * mx
        aligned = alpha.detach() * target[valid] + beta.detach()
        loss = F.smooth_l1_loss(pred[valid], aligned, reduction="none")
        losses.append((loss * w).sum() / sw)
        alphas.append(alpha.detach().to(dtype=pred.dtype))

    alpha_tensor = torch.stack(alphas).to(device=D_full.device, dtype=D_full.dtype) if alphas else D_full.new_ones((D_full.shape[0],))
    if not losses:
        return D_full.new_tensor(0.0), alpha_tensor
    return torch.stack(losses).mean().to(dtype=D_full.dtype), alpha_tensor


def boundary_ordinal_loss(
    D_full: torch.Tensor,
    R_G: torch.Tensor,
    C_G: torch.Tensor,
    rgb: torch.Tensor,
    alpha: torch.Tensor | None = None,
    conf_threshold: float = 0.05,
    rgb_tau: float = 0.04,
    geom_tau: float = 0.20,
) -> torch.Tensor:
    """Ordinal boundary supervision with SSI-fit orientation correction."""
    log_d = torch.log(D_full.clamp_min(1e-3))
    if alpha is None:
        orientation = torch.ones((D_full.shape[0], 1, 1, 1), device=D_full.device, dtype=D_full.dtype)
    else:
        sign = torch.sign(alpha).view(-1, 1, 1, 1)
        orientation = torch.where(sign == 0, torch.ones_like(sign), sign).to(dtype=D_full.dtype)
    R = R_G * orientation

    total = D_full.new_tensor(0.0)
    denom = D_full.new_tensor(0.0)

    dR_x = R[..., :, 1:] - R[..., :, :-1]
    dD_x = log_d[..., :, 1:] - log_d[..., :, :-1]
    dI_x = (rgb[..., :, 1:] - rgb[..., :, :-1]).abs().mean(dim=1, keepdim=True)
    C_x = torch.minimum(C_G[..., :, 1:], C_G[..., :, :-1]).clamp(0.0, 1.0)
    y_x = torch.sign(dR_x)
    m_x = ((dI_x > rgb_tau) | (dR_x.abs() > geom_tau)) & (C_x > conf_threshold) & (y_x.abs() > 0)
    w_x = C_x * m_x.float()
    total = total + (F.softplus(-y_x * dD_x) * w_x).sum()
    denom = denom + w_x.sum()

    dR_y = R[..., 1:, :] - R[..., :-1, :]
    dD_y = log_d[..., 1:, :] - log_d[..., :-1, :]
    dI_y = (rgb[..., 1:, :] - rgb[..., :-1, :]).abs().mean(dim=1, keepdim=True)
    C_y = torch.minimum(C_G[..., 1:, :], C_G[..., :-1, :]).clamp(0.0, 1.0)
    y_y = torch.sign(dR_y)
    m_y = ((dI_y > rgb_tau) | (dR_y.abs() > geom_tau)) & (C_y > conf_threshold) & (y_y.abs() > 0)
    w_y = C_y * m_y.float()
    total = total + (F.softplus(-y_y * dD_y) * w_y).sum()
    denom = denom + w_y.sum()

    if float(denom.detach().cpu()) < 1.0:
        return D_full.new_tensor(0.0)
    return total / denom.clamp_min(1e-6)


def scheduled_weight(target: float, epoch: int, start_epoch: int, ramp_epochs: int) -> float:
    if epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return float(target)
    progress = min(1.0, max(0.0, (epoch - start_epoch + 1) / float(ramp_epochs)))
    return float(target) * progress


def range_balanced_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    bins: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...],
) -> torch.Tensor:
    if len(bins) < 2:
        return pred.new_tensor(0.0)
    losses: list[torch.Tensor] = []
    active_weights: list[float] = []
    for idx in range(len(bins) - 1):
        lo = float(bins[idx])
        hi = float(bins[idx + 1])
        w = float(weights[min(idx, len(weights) - 1)]) if weights else 1.0
        bin_mask = mask & (target >= lo) & (target < hi)
        if int(bin_mask.sum().detach().cpu()) < 1:
            continue
        losses.append(w * mean_valid(huber(pred - target), bin_mask))
        active_weights.append(w)
    if not losses:
        return pred.new_tensor(0.0)
    return torch.stack(losses).sum() / max(1e-6, float(sum(active_weights)))


def _tensor_mean_or_zero(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if int(mask.sum().detach().cpu()) < 1:
        return value.new_tensor(0.0)
    return value[mask].mean()


def geort_loss(
    pred: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    loss_cfg: dict[str, Any],
    schedule_cfg: dict[str, Any],
    epoch: int,
    mono_ssi_cfg: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the theory-aligned full-resolution GeoRT objective."""
    D_full = pred.get("D_full", pred["D_c"])
    C_full = pred.get("C_full")
    D_1_4 = pred.get("D_1_4", pred["D_c"])
    if C_full is None:
        C_full = F.interpolate(pred.get("C", torch.ones_like(D_1_4)), size=D_full.shape[-2:], mode="bilinear", align_corners=False)
    C_full = C_full.clamp(1e-4, 1.0)

    max_depth = float(loss_cfg.get("max_depth", 120.0))
    min_depth = float(loss_cfg.get("min_depth", 1e-3))
    gt = batch["gt"]
    sparse = batch["sparse"]
    gt_mask = (batch["gt_mask"] > 0.5) & (gt > min_depth) & (gt < max_depth) & torch.isfinite(gt)
    sparse_mask = (batch["mask"] > 0.5) & (sparse > min_depth) & (sparse < max_depth) & torch.isfinite(sparse)

    if "D_cm" in batch:
        D_cm = match_prediction_size(batch["D_cm"].float(), D_full.shape[-2:], mode="bilinear")
        C_cm = match_prediction_size(batch.get("C_cm", torch.ones_like(D_cm)).float(), D_full.shape[-2:], mode="bilinear").clamp(0.0, 1.0)
    elif "D_teacher" in batch:
        D_cm = match_prediction_size(batch["D_teacher"].float(), D_full.shape[-2:], mode="bilinear")
        C_cm = match_prediction_size(batch.get("C_teacher", torch.ones_like(D_cm)).float(), D_full.shape[-2:], mode="bilinear").clamp(0.0, 1.0)
    else:
        D_cm = torch.zeros_like(D_full)
        C_cm = torch.zeros_like(D_full)
    cm_mask = (C_cm > 0.0) & (D_cm > min_depth) & (D_cm < max_depth) & torch.isfinite(D_cm)
    cm_non_gt_mask = cm_mask & ~gt_mask

    L_gt = mean_valid(huber(D_full - gt), gt_mask)
    L_S = mean_valid((D_full - sparse).abs(), sparse_mask)

    add_teacher_epoch = int(schedule_cfg.get("add_teacher_epoch", 5))
    add_geometry_epoch = int(schedule_cfg.get("add_geometry_epoch", 10))
    add_conf_epoch = int(schedule_cfg.get("add_confidence_epoch", 15))
    teacher_ramp_epochs = int(schedule_cfg.get("teacher_ramp_epochs", 5))
    geometry_ramp_epochs = int(schedule_cfg.get("geometry_ramp_epochs", 5))
    confidence_ramp_epochs = int(schedule_cfg.get("confidence_ramp_epochs", 3))

    w_cm = scheduled_weight(float(loss_cfg.get("lambda_cm", loss_cfg.get("lambda_T", 0.4))), epoch, add_teacher_epoch, teacher_ramp_epochs)
    w_ssi = scheduled_weight(float(loss_cfg.get("lambda_ssi", (mono_ssi_cfg or {}).get("weight", 0.0))), epoch, add_geometry_epoch, geometry_ramp_epochs)
    w_ord = scheduled_weight(float(loss_cfg.get("lambda_ord", 0.03)), epoch, add_geometry_epoch, geometry_ramp_epochs)
    w_C = scheduled_weight(float(loss_cfg.get("lambda_C", 0.05)), epoch, add_conf_epoch, confidence_ramp_epochs)

    if epoch >= add_teacher_epoch:
        min_dense_pixels = int(loss_cfg.get("min_dense_metric_pixels", 1024))
        if bool(loss_cfg.get("require_dense_metric_teacher", True)) and int(cm_non_gt_mask.sum().detach().cpu()) < min_dense_pixels:
            raise RuntimeError(
                "Dense metric teacher is missing for this batch: D_cm/C_cm contains too few non-GT pixels. "
                "Generate DMD3C or metric_coarse outputs for the train split before training past add_teacher_epoch."
            )
        cm_weight = C_cm
        if bool(loss_cfg.get("metric_teacher_range_weight", True)):
            range_scale = float(loss_cfg.get("metric_teacher_range_scale", 0.5))
            cm_weight = cm_weight * (1.0 + range_scale * (D_cm / max_depth).clamp(0.0, 1.0))
        L_T = mean_valid(cm_weight * huber(D_full - D_cm), cm_mask)
    else:
        L_T = D_full.new_tensor(0.0)

    scale = max(1, batch["sparse"].shape[-2] // D_1_4.shape[-2])
    gt_ds, gt_mask_ds = downsample_depth_with_mask(gt, gt_mask.float(), scale=scale)
    D_cm_ds, C_cm_ds = downsample_depth_with_conf(D_cm, C_cm * cm_mask.float(), scale=scale)
    gt_ds = match_prediction_size(gt_ds, D_1_4.shape[-2:])
    gt_mask_ds = match_prediction_size(gt_mask_ds, D_1_4.shape[-2:])
    D_cm_ds = match_prediction_size(D_cm_ds, D_1_4.shape[-2:], mode="bilinear")
    C_cm_ds = match_prediction_size(C_cm_ds, D_1_4.shape[-2:], mode="bilinear")
    L_gt_1_4 = mean_valid(huber(D_1_4 - gt_ds), gt_mask_ds > 0.5)
    L_cm_1_4 = mean_valid(C_cm_ds * huber(D_1_4 - D_cm_ds), C_cm_ds > 0.0) if epoch >= add_teacher_epoch else D_full.new_tensor(0.0)
    L_aux = float(loss_cfg.get("lambda_gt_1_4", 1.0)) * L_gt_1_4 + float(loss_cfg.get("lambda_cm_1_4", 1.0)) * L_cm_1_4
    L_range = range_balanced_loss(
        D_full,
        gt,
        gt_mask,
        loss_cfg.get("range_bins", [0.0, 20.0, 40.0, 60.0, 80.0, 120.0]),
        loss_cfg.get("range_weights", [1.0, 1.2, 1.5, 2.0, 2.5]),
    )

    L_ssi = D_full.new_tensor(0.0)
    L_ord = D_full.new_tensor(0.0)
    geom_valid_pixels = D_full.new_tensor(0.0)
    if epoch >= add_geometry_epoch:
        if "R_G" in batch and "C_G" in batch:
            R_G = match_prediction_size(batch["R_G"].float(), D_full.shape[-2:], mode="bilinear")
            C_G = match_prediction_size(batch["C_G"].float(), D_full.shape[-2:], mode="bilinear")
            geom_mask = (C_G > float(loss_cfg.get("geometry_conf_threshold", 0.05))) & torch.isfinite(R_G)
            geom_valid_pixels = geom_mask.float().sum()
            if bool(loss_cfg.get("require_geometry_teacher", True)) and int(geom_valid_pixels.detach().cpu()) < int(loss_cfg.get("geometry_min_valid_pixels", 512)):
                raise RuntimeError(
                    "Geometry teacher is missing for this batch: R_G/C_G has too few valid pixels. "
                    "Generate geometry_fused outputs or Depth Anything raw maps before training past add_geometry_epoch."
                )
            L_ssi, alpha = geometry_ssi_loss(
                D_full,
                R_G,
                C_G,
                conf_threshold=float(loss_cfg.get("geometry_conf_threshold", 0.05)),
                min_valid_pixels=int(loss_cfg.get("geometry_min_valid_pixels", 512)),
            )
            L_ord = boundary_ordinal_loss(
                D_full,
                R_G,
                C_G,
                batch["rgb"],
                alpha=alpha,
                conf_threshold=float(loss_cfg.get("geometry_conf_threshold", 0.05)),
            )
        elif mono_ssi_cfg and bool(mono_ssi_cfg.get("enabled", False)) and str(mono_ssi_cfg.get("key", "D_da_raw")) in batch:
            mono_key = str(mono_ssi_cfg.get("key", "D_da_raw"))
            mono = match_prediction_size(batch[mono_key].float(), D_full.shape[-2:], mode="bilinear")
            mono_mask = batch.get("da_raw_valid", torch.ones_like(mono))
            mono_mask = match_prediction_size(mono_mask.float(), D_full.shape[-2:], mode="nearest")
            geom_valid_pixels = ((mono_mask > 0.5) & torch.isfinite(mono)).float().sum()
            if bool(loss_cfg.get("require_geometry_teacher", True)) and int(geom_valid_pixels.detach().cpu()) < int(mono_ssi_cfg.get("min_valid_pixels", 512)):
                raise RuntimeError(
                    "Mono SSI teacher is missing for this batch. Generate Depth Anything raw maps for the train split "
                    "or disable require_geometry_teacher for debugging."
                )
            L_ssi, _ = geometry_ssi_loss(D_full, mono, mono_mask, min_valid_pixels=int(mono_ssi_cfg.get("min_valid_pixels", 512)))
        elif bool(loss_cfg.get("require_geometry_teacher", True)) and (w_ssi > 0.0 or w_ord > 0.0):
            raise RuntimeError(
                "Geometry distillation is scheduled but neither R_G/C_G nor the configured mono SSI map is present in the batch."
            )

    L_C_reg = D_full.new_tensor(0.0)
    L_C_calib = D_full.new_tensor(0.0)
    if epoch >= add_conf_epoch:
        D_sup = torch.where(gt_mask, gt, D_cm)
        sup_mask = gt_mask | cm_mask
        s_full = -torch.log(C_full.clamp_min(1e-6))
        conf_term = C_full * huber(D_full - D_sup) + float(loss_cfg.get("lambda_s", 0.01)) * s_full
        L_C_reg = mean_valid(conf_term, sup_mask)
        tau = float(loss_cfg.get("confidence_tau", 2.0))
        c_target = torch.exp(-(D_full.detach() - D_sup).abs() / max(tau, 1e-6)).clamp(0.0, 1.0)
        L_C_calib = mean_valid((C_full - c_target).abs(), sup_mask)
        L_C = L_C_reg + float(loss_cfg.get("lambda_C_calib", 0.5)) * L_C_calib
    else:
        L_C = D_full.new_tensor(0.0)

    L_E = edge_smoothness_full(D_full, batch["rgb"], max_depth=max_depth)

    total = (
        float(loss_cfg.get("lambda_gt", 1.0)) * L_gt
        + float(loss_cfg.get("lambda_S", 1.0)) * L_S
        + w_cm * L_T
        + float(loss_cfg.get("lambda_aux", 0.2)) * L_aux
        + float(loss_cfg.get("lambda_range", 0.15)) * L_range
        + w_ssi * L_ssi
        + w_ord * L_ord
        + w_C * L_C
        + float(loss_cfg.get("lambda_E", 0.01)) * L_E
    )
    full_pixels = float(D_full.numel())
    cm_valid_ratio = float(cm_mask.float().mean().detach().cpu())
    cm_non_gt_ratio = float(cm_non_gt_mask.float().mean().detach().cpu())
    cm_conf_mean = float(_tensor_mean_or_zero(C_cm, cm_mask).detach().cpu())
    dcm_gt_abs = float(_tensor_mean_or_zero((D_cm - gt).abs(), cm_mask & gt_mask).detach().cpu())
    geom_valid_ratio = float((geom_valid_pixels / max(1.0, full_pixels)).detach().cpu()) if torch.is_tensor(geom_valid_pixels) else 0.0
    items = {
        "loss": float(total.detach().cpu()),
        "L_gt": float(L_gt.detach().cpu()),
        "L_T": float(L_T.detach().cpu()),
        "L_S": float(L_S.detach().cpu()),
        "L_aux": float(L_aux.detach().cpu()),
        "L_range": float(L_range.detach().cpu()),
        "L_C": float(L_C.detach().cpu()),
        "L_C_reg": float(L_C_reg.detach().cpu()),
        "L_C_calib": float(L_C_calib.detach().cpu()),
        "L_E": float(L_E.detach().cpu()),
        "L_ssi": float(L_ssi.detach().cpu()),
        "L_ord": float(L_ord.detach().cpu()),
        "w_cm": w_cm,
        "w_ssi": w_ssi,
        "w_ord": w_ord,
        "w_C": w_C,
        "cm_valid_ratio": cm_valid_ratio,
        "cm_non_gt_ratio": cm_non_gt_ratio,
        "cm_conf_mean": cm_conf_mean,
        "cm_gt_abs": dcm_gt_abs,
        "geom_valid_ratio": geom_valid_ratio,
    }
    return total, items
