"""Continuation objective: GT priority, output RMSE, cautious KD, noisy-sensor trust."""
import torch
from torch.nn import functional as F
from loss_helpers import pooled, mean_masked, huber, pair_gradient, teacher_weights, range_rmse


def sensor_target(batch):
    # Used ONLY by the training objective, never by the model/evaluator.
    gt, sparse = batch["gt"].float(), batch["sparse"].float()
    truth = torch.exp(-(sparse - gt).abs() / (.25 + .01 * gt.clamp_min(.1)))
    return torch.where(batch["gt_mask"] > 0, truth, torch.ones_like(truth))


def objective(pred, batch, holdout_mask, epoch_progress, config):
    gt, valid = batch["gt"].float(), batch["gt_mask"].float()
    final, raw = pred["D_full"].float(), pred["D1"].float()
    zero = final.sum() * 0
    parts = {}
    metric = zero
    for name, coefficient in (("D16", .025), ("D8", .05), ("D4", .15), ("D2", .30), ("D1", .50), ("D_full", 1.)):
        target, support = pooled(gt, valid, pred[name].shape[-2:])
        term = mean_masked(huber(pred[name].float() - target), (support > 0).float())
        parts[f"gt_{name}"] = term
        metric = metric + coefficient * term
    rmse = (mean_masked((final - gt).square(), valid) + 1e-6).sqrt() - .001
    log = mean_masked(huber(final.clamp_min(.1).log() - gt.clamp_min(.1).log(), .1), valid)
    edge = pair_gradient(final.clamp_min(.1).log(), gt.clamp_min(.1).log(), valid)
    balanced, active_bins = range_rmse(final, gt, valid, config.get("range_min_pixels", 64))
    quality = sensor_target(batch)
    observed = batch["mask"] * (1 - holdout_mask)
    # Normalize by observed count, so downweighting conflicts is not cancelled.
    sparse = (huber(raw - batch["sparse"]) * observed * quality).sum() / observed.sum().clamp_min(1)
    holdout = (huber(raw - batch["sparse"]) * holdout_mask * quality).sum() / holdout_mask.sum().clamp_min(1)
    trust_mask = observed * valid
    trust_weights = 1 + 3 * (1 - quality)
    trust = (F.binary_cross_entropy_with_logits(pred["sensor_logits"].float(), quality, reduction="none")
             * trust_mask * trust_weights).sum() / (trust_mask * trust_weights).sum().clamp_min(1)
    kd, kd_edge, coverage, mean_conf = zero, zero, zero, zero
    fraction = min(1., max(0., float(epoch_progress) / max(1, config["epochs"])))
    kd_weight = config["kd_weight"] * (1 - .5 * fraction) if config["teacher_enabled"] else 0.
    if config["teacher_enabled"]:
        teacher, confidence, forbidden, eligible = teacher_weights(batch, config["kd_conf_min"])
        coverage = eligible.float().mean()
        mean_conf = confidence.sum() / eligible.sum().clamp_min(1)
        for name, coefficient in (("D4", .25), ("D2", .50), ("D1", .25)):
            depth = pred[name].float()
            target, weight = pooled(teacher, confidence, depth.shape[-2:])
            factor = gt.shape[-1] // depth.shape[-1]
            blocked = F.max_pool2d(forbidden, factor, factor) if factor > 1 else forbidden
            weight = weight * (blocked == 0)
            term = huber(depth - target) + .2 * huber(depth.clamp_min(.1).log() - target.clamp_min(.1).log(), .1)
            kd = kd + coefficient * (term * weight).sum() / (weight > 0).sum().clamp_min(1)
            if name == "D2":
                kd_edge = pair_gradient(depth.clamp_min(.1).log(), target.clamp_min(.1).log(), weight)
    weighted = {"metric": metric, "rmse": config["rmse_weight"] * rmse,
                "range": config["range_weight"] * balanced, "log": .2 * log, "edge": .05 * edge,
                "sparse": .02 * sparse, "holdout": .05 * holdout, "trust": .02 * trust,
                "metric_kd": kd_weight * kd,
                "teacher_edge": (config["kd_edge_weight"] if config["teacher_enabled"] else 0) * kd_edge}
    total = sum(weighted.values())
    stats = {**parts, "total": total, "metric": metric, "rmse": rmse, "range": balanced,
             "range_active_bins": active_bins, "sparse": sparse, "holdout": holdout, "trust": trust,
             "metric_kd": kd, "kd_coverage": coverage, "kd_mean_confidence": mean_conf,
             "lambda_kd": zero.detach() + kd_weight,
             "sensor_gate_mean": pred["sensor_gate"].sum() / observed.sum().clamp_min(1),
             "sensor_gt_conflict_fraction": ((quality < .5) * trust_mask).sum() / trust_mask.sum().clamp_min(1),
             "abs_delta4_mean": pred["delta4"].abs().mean(), "abs_delta1_mean": pred["delta1"].abs().mean(),
             **{f"weighted_{key}": value for key, value in weighted.items()}}
    return total, {key: value.detach() for key, value in stats.items()}
