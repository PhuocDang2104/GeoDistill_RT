"""Deployment-copy-only algebraic rewrites. No new learned layer or training path."""
import torch
from torch.nn import functional as F
from core import PhaseUpsample
from model_parent import ContextPhaseUpsample
from model_base import InnovationReadout
from phase_readout import PhaseInnovationResidual
from geometry_primitives import candidate_queries
from geometry_primitives import adjacent,edge_weights
from geometry import JetNODE,decode_jet
from device_constants import device_constant


def neighbours9(depth):
    """Same row-major one-hot 3x3 Conv stencil, including replicate boundaries."""
    h,w=depth.shape[-2:]
    p=F.pad(depth.float(),(1,1,1,1),mode='replicate')
    return torch.cat(tuple(p[...,y:y+h,x:x+w] for y in range(3) for x in range(3)),1)


def pooled7(x):
    # Keep original 2D accumulation order: separable pooling amplified FP32
    # cancellation in low-support variance and was REJECTED in local testing.
    return F.avg_pool2d(x,7,1,3)


def cardinal_translate(j):
    """Same quadratic origin shifts for E/W/S/N; omit known zero products.

    j is B,4,6,H,W, direction order matches adjacent(). No approximation to ODE.
    """
    result=[]
    for index in range(4):
        v,gx,gy,hxx,hxy,hyy=j[:,index].split(1,1)
        if index==0:values=(v-gx+.5*hxx,gx-hxx,gy-hxy,hxx,hxy,hyy)
        elif index==1:values=(v+gx+.5*hxx,gx+hxx,gy+hxy,hxx,hxy,hyy)
        elif index==2:values=(v-gy+.5*hyy,gx-hxy,gy-hyy,hxx,hxy,hyy)
        else:values=(v+gy+.5*hyy,gx+hxy,gy+hyy,hxx,hxy,hyy)
        result.append(torch.cat(values,1))
    return torch.stack(result,1)


class FastJetNODE(JetNODE):
    def vector_field(self,t,z,context,barrier,j0,state):
        with torch.autocast(z.device.type,enabled=False):
            j=decode_jet(z.float());mean,valid,density,_,_,spread=state
            v=j[:,:1];depth=v.reciprocal();v0=j0[:,:1]
            reliability=valid*torch.exp(-4*spread)
            innovation=valid*(mean.clamp_min(.1).reciprocal()-v)/v
            metric_error=valid*(mean-depth)/20
            tau=t.float().reshape(1,1,1,1).expand_as(v)
            inputs=torch.cat((j/v0,v/v0-1,metric_error.clamp(-4,4),innovation.clamp(-2,2),
                valid,density,spread.clamp_max(2),reliability,tau),1).float()
            hidden=self.update(context.float()+self.state(inputs))
            forcing=self.reaction(hidden).tanh()*v*self.forcing_limits.view(1,6,1,1)
            source=self.sensor(hidden).sigmoid()*reliability*innovation.clamp(-1,1)*v
            forcing=forcing+torch.cat((source,torch.zeros_like(j[:,1:])),1)
            rates=.6*self.conductance(hidden).sigmoid()*(1-barrier.float().sigmoid())
            neighbour_v=adjacent(v)
            mismatch=(neighbour_v-v[:,None]).abs()/(neighbour_v*v[:,None]).clamp_min(1e-6)
            tolerance=.5+.015*(depth[:,None]+neighbour_v.reciprocal())
            weights=edge_weights(rates)*(.05+.95*torch.exp(-mismatch/tolerance))
            transport=(weights*(cardinal_translate(adjacent(j))-j[:,None])).sum(1)
            scales=device_constant(z,[1.,.5,.5,.25,.25,.25]).view(1,6,1,1)
            normalized=(forcing+transport)/(v*scales)
            return 2*torch.tanh(normalized/2)


def sparse_innovation(base,sparse,mask):
    error=mask.float()*(sparse.float()-base.float())
    mass,numerator,second=pooled7(torch.cat((mask.float(),error,error.square()),1)).split(1,1)
    denominator=mass.clamp_min(1e-6);mean=numerator/denominator
    support=(mass>0).float()
    std=(((second/denominator-mean.square()).clamp_min(0)+1e-6).sqrt()-.001)*support
    return torch.cat((mean/20,std/20,mass),1)


def phase_innovation(depth,sparse,mask):
    d,s,m=depth.float(),sparse.float(),mask.float()
    valid=F.pixel_unshuffle(m,2);err=F.pixel_unshuffle(m*(s-d),2)
    mass,numerator,second=F.avg_pool2d(torch.cat((valid,err,err.square()),1),3,1,1).split(4,1)
    denominator=mass.clamp_min(1e-6);mean=numerator/denominator
    support=(mass>0).float()
    std=((second/denominator-mean.square()).clamp_min(0)+1e-6).sqrt()-.001
    packed_log=F.pixel_unshuffle(d.clamp_min(.1).log(),2)
    contrast=packed_log-packed_log.mean(1,keepdim=True)
    return torch.cat(((mean/20).clamp(-6,6)*support,(std/20).clamp(0,6)*support,
                      mass,contrast.clamp(-4,4)),1),support


class FastUpsample(PhaseUpsample):
    def forward(self,depth,features):
        hidden=self.trunk(features)
        weights=self.weights(hidden).float().reshape(depth.shape[0],4,9,*depth.shape[-2:]).softmax(2)
        value=(weights*neighbours9(depth)[:,None]).sum(2)
        phases=value+(self.residual_base+self.residual_ratio*value)*self.residual(hidden).float().tanh()
        return F.pixel_shuffle(phases,2).clamp(.1,120)


class FastContextUpsample(ContextPhaseUpsample):
    def forward(self,depth,features,rich=None):
        hidden=self.trunk(features)
        if rich is not None:hidden=hidden+rich[:,:16]
        weights=self.weights(hidden).float().reshape(depth.shape[0],4,9,*depth.shape[-2:]).softmax(2)
        value=(weights*neighbours9(depth)[:,None]).sum(2)
        phases=value+(self.residual_base+self.residual_ratio*value)*self.residual(hidden).float().tanh()
        return F.pixel_shuffle(phases,2).clamp(.1,120)


class FastReadout(InnovationReadout):
    def forward(self,j,context,g4,base,sparse,mask,barrier,p4):
        sensor=F.pixel_unshuffle(torch.cat((base/120,mask,mask*(sparse-base)/20,sparse/120),1),2)
        hidden=self.body(torch.cat((context,g4,sensor),1))
        inv,valid=candidate_queries(j)
        bphase=F.pixel_unshuffle(base.float(),2)[:,None]
        penalty=((inv.reciprocal()-bphase).abs()/(1+.1*bphase)).clamp_max(6)
        b=barrier.float().sigmoid();east,south=b[:,:1],b[:,1:2]
        west=F.pad(east[...,:-1],(1,0,0,0));north=F.pad(south[...,:-1,:],(0,0,1,0))
        edge_cost=torch.stack((torch.zeros_like(east),east,west,south,north),1)
        logits=4*torch.tanh(self.candidates(hidden).float().reshape(j.shape[0],5,4,*j.shape[-2:])/4)
        weights=(logits-penalty-2*edge_cost+(1-valid)*(-10000.)).softmax(1)
        consensus=(weights*inv).sum(1)
        uncertainty=(weights*(inv-consensus[:,None]).square()).sum(1)/consensus.square().clamp_min(1e-6)
        query=F.pixel_shuffle(consensus.reciprocal(),2)
        strength=F.softplus(self.log_uncertainty_strength).view(1,4,1,1)
        gate=F.pixel_shuffle((self.blend(hidden).float()-strength*torch.log1p(uncertainty)).sigmoid(),2)
        geometric=gate*(query-base).clamp(-(1+.1*base),1+.1*base)
        innovation=sparse_innovation(base,sparse,mask)
        correction=self.metric(torch.cat((hidden,p4,F.pixel_unshuffle(innovation,2),uncertainty),1)).float()
        raw=self.delta(hidden).float()+correction
        delta=(1+.05*base)*F.pixel_shuffle(raw.tanh(),2)
        return (base+geometric+delta).clamp(.1,120),query,geometric,delta,{
            'query_uncertainty':F.pixel_shuffle(uncertainty,2)}


class FastPIR(PhaseInnovationResidual):
    def forward(self,hidden,depth,sparse,mask,jet_uncertainty=None):
        features,support=phase_innovation(depth,sparse,mask)
        inputs=torch.cat((hidden,features),1)
        if self.uncertainty_conditioned:
            if jet_uncertainty is None:raise ValueError('UPIR requires consensus uncertainty')
            u=jet_uncertainty.float().clamp_min(0)
            inputs=torch.cat((inputs,u/(u+self.uncertainty_scale)),1)
        output=self.head(self.body(inputs)).float()
        delta,gates=output[:,:4],output[:,4:].sigmoid()
        return F.pixel_shuffle(delta.tanh()*gates,2)*(.5+.05*depth.float()),gates,support


def enable_fast_ops(model):
    if model.training:raise ValueError('Fast operators are deployment only')
    if model.phase2.diagnostics:raise ValueError('Disable telemetry before deployment rewrite')
    # Per-instance class changes; NEVER mutate shared module globals or trained tensors.
    model.up4_2.__class__=FastUpsample
    model.up2_1.__class__=FastContextUpsample
    model.phase2.__class__=FastReadout
    model.detail1.pir.__class__=FastPIR
    model.dynamics.__class__=FastJetNODE
    return model
