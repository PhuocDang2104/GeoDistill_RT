"""GT-first refinement: robust excess risk, weak structural KD, no ordinal sorting."""
import torch
from torch.nn import functional as F
from losses_v8 import objective as baseline_objective
from loss_helpers import pooled,mean_masked,huber


def local_normalize(x,valid,kernel=17):
    w=F.avg_pool2d(valid,kernel,1,kernel//2)
    mean=F.avg_pool2d(x*valid,kernel,1,kernel//2)/w.clamp_min(1e-6)
    var=F.avg_pool2d(x.square()*valid,kernel,1,kernel//2)/w.clamp_min(1e-6)-mean.square()
    return (x-mean)/var.clamp_min(1e-6).sqrt(),(w>.25).float()*valid


def relative_structure(depth,relative,confidence,rgb,forbidden):
    shape=depth.shape[-2:]; relative,c=pooled(relative,confidence,shape)
    factor=forbidden.shape[-1]//shape[-1]
    blocked=F.max_pool2d(forbidden,factor,factor) if factor>1 else forbidden
    eligible=(c>=.35).float()*(blocked==0)
    pred,support=local_normalize(depth.float().clamp_min(.1).reciprocal(),eligible)
    target,support_t=local_normalize(relative.float(),eligible); support=support*support_t
    grey=F.interpolate(rgb.float().mean(1,keepdim=True),size=shape,mode="area")
    gradient,coverage=depth.sum()*0,depth.sum()*0
    for offset in (1,4):
        for axis in (-1,-2):
            if axis==-1: a,b=(...,slice(None),slice(offset,None)),(...,slice(None),slice(None,-offset))
            else: a,b=(...,slice(offset,None),slice(None)),(...,slice(None,-offset),slice(None))
            dp,dt=pred[a]-pred[b],target[a]-target[b]
            pair=support[a]*support[b]*torch.minimum(c[a],c[b])
            weight=pair*((dt.abs()>.05)|((grey[a]-grey[b]).abs()>.05)).float()
            gradient+=(huber(dp-dt,.25)*weight).sum()/(weight>0).sum().clamp_min(1)
            coverage+=(weight>0).float().mean()
    # Compatibility return schema; ordinal is deliberately disabled, not computed.
    return gradient/4,depth.sum()*0,coverage/4


def robust_tail(residual,valid,threshold=2.,delta=10.):
    excess=F.relu(residual.float().abs()-threshold)
    # Quadratic below delta, linear beyond; never a hard cap with zero gradient.
    return mean_masked(2*huber(excess,delta),valid.float())


def objective(pred,batch,holdout_mask,epoch_progress,config):
    if config.get("model_name")=="v8_control":
        return baseline_objective(pred,batch,holdout_mask,epoch_progress,config)
    original={**config,"tail_weight":0}
    total,stats=baseline_objective(pred,batch,holdout_mask,epoch_progress,original)
    zero=pred["D_full"].sum()*0
    tail=robust_tail(pred["D_full"]-batch["gt"],batch["gt_mask"],
                     config.get("tail_threshold_m",2.),config.get("tail_huber_delta_m",10.))
    tail_ramp=min(1.,max(0.,float(epoch_progress))/max(1e-6,config.get("tail_warmup_epochs",2.)))
    weighted_tail=config.get("tail_weight",.25)*tail_ramp*tail
    inverse=zero
    for name,weight in (("D4",.25),("D2",.5),("D_full",1.)):
        target,support=pooled(batch["gt"].float(),batch["gt_mask"].float(),pred[name].shape[-2:])
        error=100*(pred[name].float().clamp_min(.1).reciprocal()-target.clamp_min(.1).reciprocal())
        inverse+=weight*mean_masked(huber(error,1.),(support>0).float())
    weighted_inverse=config.get("inverse_weight",.01)*inverse
    grad,ordinal,coverage=zero,zero,zero
    if config.get("relative_enabled",True):
        if not {"relative","relative_confidence"}.issubset(batch):
            raise RuntimeError("V9.1 requires audited relative cache; no fallback")
        forbidden=((batch["gt_mask"]>0)|(batch["mask"]>0)).float()
        grad,ordinal,coverage=relative_structure(pred["D2"],batch["relative"],batch["relative_confidence"],batch["rgb"],forbidden)
    ramp=min(1.,max(0.,float(epoch_progress))/max(1e-6,config.get("relative_warmup_epochs",3.)))
    weighted_relative=config.get("relative_weight",.015)*ramp*grad
    total=total+weighted_tail+weighted_inverse+weighted_relative
    stats.update(total=total.detach(),tail=tail.detach(),tail_ramp=zero.detach()+tail_ramp,
                 weighted_tail=weighted_tail.detach(),inverse=inverse.detach(),weighted_inverse=weighted_inverse.detach(),
                 relative_gradient=grad.detach(),relative_ordinal=ordinal.detach(),relative_pair_coverage=coverage.detach(),
                 relative_ramp=zero.detach()+ramp,weighted_relative=weighted_relative.detach())
    for key in ("query_uncertainty_mean","query_gate_mean","query_center_weight_mean","query_entropy_mean",
                "innovation_abs_mean_m","innovation_support_fraction","innovation_head_raw_abs","metric_head_saturation_fraction"):
        stats[key]=pred.get(key,zero).detach()
    return total,stats
