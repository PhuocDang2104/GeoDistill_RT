import torch
from torch.nn import functional as F
from loss_helpers import pooled,huber
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

