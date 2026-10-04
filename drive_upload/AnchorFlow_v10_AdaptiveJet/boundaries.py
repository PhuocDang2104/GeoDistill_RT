"""Observed GT discontinuities, not RGB texture edges or dense occlusion labels.

Fixed thresholds, valid paired GT only. Chebyshev-radius bands by max-pooling.
Masks never depend on predictions; no GT/teacher is an inference input.
"""
import torch
from torch.nn import functional as F

BOUNDARY_PROTOCOL = {"absolute_jump_m":1.,"relative_jump":.05,
                     "bands_chebyshev_px":[1,2,3,5,10],
                     "quarter_max_relative_std":.05,"quarter_min_gt_count":2}


def pair_targets(gt,valid):
    gt = torch.where(torch.isfinite(gt)&(valid>0),gt.float(),torch.zeros_like(gt,dtype=torch.float32))
    valid = (valid>0)&(gt>.1)&(gt<120)
    hvalid = valid[...,1:]&valid[...,:-1]
    vvalid = valid[...,1:,:]&valid[...,:-1,:]
    h = (gt[...,1:]-gt[...,:-1]).abs() > torch.maximum(torch.ones_like(gt[...,1:]),.05*torch.minimum(gt[...,1:],gt[...,:-1]))
    v = (gt[...,1:,:]-gt[...,:-1,:]).abs() > torch.maximum(torch.ones_like(gt[...,1:,:]),.05*torch.minimum(gt[...,1:,:],gt[...,:-1,:]))
    return torch.cat((F.pad((h&hvalid).float(),(0,1,0,0)),F.pad((v&vvalid).float(),(0,0,0,1))),1), \
           torch.cat((F.pad(hvalid.float(),(0,1,0,0)),F.pad(vvalid.float(),(0,0,0,1))),1)


def boundary_mask(gt,valid):
    labels,_ = pair_targets(gt,valid)
    h,v = labels.split(1,1)
    return (h+v+F.pad(h[...,:-1],(1,0,0,0))+F.pad(v[...,:-1,:],(0,0,1,0)))>0


def band_mask(edges,radius):
    return F.max_pool2d(edges.float(),2*radius+1,1,radius)>0


def quarter_targets(gt,valid):
    # Reject mixed-surface cells; pooled GT means near edges are not clean labels.
    m = ((valid>0)&torch.isfinite(gt)&(gt>.1)&(gt<120)).float()
    g = torch.where(m>0,gt.float(),torch.zeros_like(gt,dtype=torch.float32))
    density = F.avg_pool2d(m,4,4)
    mean = F.avg_pool2d(g,4,4)/density.clamp_min(1e-6)
    second = F.avg_pool2d(g.square(),4,4)/density.clamp_min(1e-6)
    spread = (second-mean.square()).clamp_min(0).sqrt()/mean.clamp_min(.1)
    clean = (density*16>=2)&(spread<=.05)
    return pair_targets(mean,clean)


def balanced_barrier_loss(logits,gt,valid):
    labels,support = quarter_targets(gt,valid)
    bce = F.binary_cross_entropy_with_logits(logits.float(),labels,reduction="none")
    pos,neg = support*labels,support*(1-labels)
    npos,nneg = pos.sum(),neg.sum()
    active = (npos>0).float()+(nneg>0).float()
    loss = ((bce*pos).sum()/npos.clamp_min(1)+(bce*neg).sum()/nneg.clamp_min(1))/active.clamp_min(1)
    return loss, support.mean(), npos/support.sum().clamp_min(1)


class BoundaryMetrics:
    def __init__(self,device):
        self.radii = (1,2,3,5,10)
        # count/SSE/SAE/tail counts for cumulative bands and disjoint rings.
        self.sums = torch.zeros(10,8,device=device,dtype=torch.float64)
        self.total_sse = torch.zeros((),device=device,dtype=torch.float64)

    def update(self,pred,gt,valid):
        m = (valid>0)&torch.isfinite(gt)&(gt>.1)&(gt<120)
        edges = boundary_mask(gt,m)
        error = torch.where(m,pred.float()-gt.float(),torch.zeros_like(gt,dtype=torch.float32))
        self.total_sse += error.square().sum(dtype=torch.float64)
        previous = torch.zeros_like(m)
        for i,r in enumerate(self.radii):
            band = band_mask(edges,r)&m
            ring = band&~previous
            previous = band
            for j,mask in ((i,band),(i+5,ring)):
                values = [mask.sum(),(error.square()*mask).sum(dtype=torch.float64),
                          (error.abs()*mask).sum(dtype=torch.float64)]
                values += [((error.abs()>t)&mask).sum() for t in (1,2,3,5,10)]
                self.sums[j] += torch.stack(values)

    def report(self):
        total = float(self.total_sse.cpu())
        result = {"protocol":BOUNDARY_PROTOCOL,"bands":{},"rings":{}}
        for i,values in enumerate(self.sums.cpu().tolist()):
            n,sse,sae,*tails = values
            group = "bands" if i<5 else "rings"
            result[group][str(self.radii[i%5])] = {"pixels":int(n),"rmse_m":(sse/n)**.5 if n else None,
                "mae_m":sae/n if n else None,"sse_fraction_global":sse/total if total else 0,
                "bad_pixel_rates":{str(t):v/n if n else None for t,v in zip((1,2,3,5,10),tails)}}
        return result
