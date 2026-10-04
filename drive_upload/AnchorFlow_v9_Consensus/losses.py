"""V8 objective + unit-scaled inverse loss and affine-invariant relative structure.

Relative teacher never supplies metric metres; GT/sensor priority excludes
conflicting structural supervision at all original observed target cells.
"""
import torch
from torch.nn import functional as F
from losses_v8 import objective as baseline_objective
from loss_helpers import pooled, mean_masked, huber


def local_normalize(x, valid, kernel=17):
    w = F.avg_pool2d(valid, kernel, 1, kernel//2)
    mean = F.avg_pool2d(x*valid, kernel, 1, kernel//2)/w.clamp_min(1e-6)
    var = F.avg_pool2d(x.square()*valid, kernel, 1, kernel//2)/w.clamp_min(1e-6)-mean.square()
    return (x-mean)/var.clamp_min(1e-6).sqrt(), (w>.25).float()*valid


def relative_structure(depth, relative, confidence, rgb, forbidden):
    shape = depth.shape[-2:]
    relative, c = pooled(relative, confidence, shape)
    factor = forbidden.shape[-1]//shape[-1]
    blocked = F.max_pool2d(forbidden, factor, factor) if factor>1 else forbidden
    eligible = (c>=.35).float()*(blocked==0)
    pred, support = local_normalize(depth.float().clamp_min(.1).reciprocal(), eligible)
    target, support_t = local_normalize(relative.float(), eligible)
    support = support*support_t
    grey = F.interpolate(rgb.float().mean(1, keepdim=True), size=shape, mode="area")
    gradient, ordinal, count = depth.sum()*0, depth.sum()*0, depth.sum()*0
    # Tensorized local pairs; no sort, random CPU sampling or learned teacher graph.
    for offset in (1, 4):
        for axis in (-1, -2):
            if axis == -1:
                a, b = (..., slice(None), slice(offset, None)), (..., slice(None), slice(None, -offset))
            else:
                a, b = (..., slice(offset, None), slice(None)), (..., slice(None, -offset), slice(None))
            dp, dt = pred[a]-pred[b], target[a]-target[b]
            pair = support[a]*support[b]*torch.minimum(c[a], c[b])
            difficult = ((dt.abs()>.05)|((grey[a]-grey[b]).abs()>.05)).float()
            weight = pair*difficult
            gradient = gradient+(huber(dp-dt,.25)*weight).sum()/(weight>0).sum().clamp_min(1)
            ordered = weight*(dt.abs()>.1)
            # Inverse relative depth: larger = nearer. Smooth margin, not BCE(sign,sign).
            ordinal = ordinal+(F.softplus(-torch.sign(dt)*dp/.5)*ordered).sum()/(ordered>0).sum().clamp_min(1)
            count = count+(ordered>0).float().mean()
    return gradient/4, ordinal/4, count/4


def objective(pred, batch, holdout_mask, epoch_progress, config):
    total, stats = baseline_objective(pred, batch, holdout_mask, epoch_progress, config)
    zero = pred["D_full"].sum()*0
    inv = zero
    for name, weight in (("D4",.25),("D2",.5),("D_full",1.)):
        target, support = pooled(batch["gt"].float(), batch["gt_mask"].float(), pred[name].shape[-2:])
        # 100 m^-1 corresponds to scaled inverse units (benchmark km^-1 / 10).
        error = 100*(pred[name].float().clamp_min(.1).reciprocal()-target.clamp_min(.1).reciprocal())
        inv = inv+weight*mean_masked(huber(error,1.), (support>0).float())
    inv_weighted = config.get("inverse_weight",.01)*inv
    grad, ordinal, coverage = zero, zero, zero
    if config.get("relative_enabled",True):
        if not {"relative","relative_confidence"}.issubset(batch):
            raise RuntimeError("V9 requires audited relative teacher cache; no fallback permitted")
        forbidden = ((batch["gt_mask"]>0)|(batch["mask"]>0)).float()
        grad, ordinal, coverage = relative_structure(pred["D2"],batch["relative"],batch["relative_confidence"],batch["rgb"],forbidden)
    ramp = min(1.,max(0.,float(epoch_progress))/max(1e-6,config.get("relative_warmup_epochs",3.)))
    relative_weighted = config.get("relative_weight",.05)*ramp*(grad+.1*ordinal)
    total = total+inv_weighted+relative_weighted
    stats.update(total=total.detach(), inverse=inv.detach(), relative_gradient=grad.detach(),
                 relative_ordinal=ordinal.detach(), relative_pair_coverage=coverage.detach(),
                 relative_ramp=zero.detach()+ramp, weighted_inverse=inv_weighted.detach(),
                 weighted_relative=relative_weighted.detach())
    for key in ("query_uncertainty_mean","query_gate_mean","query_center_weight_mean","query_entropy_mean"):
        stats[key] = pred.get(key,zero).detach()
    return total, stats
