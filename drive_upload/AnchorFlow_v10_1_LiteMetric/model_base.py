from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from core import ConvBN,LiteBlock,SparsePyramid,PyramidDecoder,PhaseUpsample
from support import Context32,PhaseDetailAndTrust
from geometry import FixedJetDynamics,candidate_queries

def sparse_innovation(base,sparse,mask):
    """Local 7x7 half-grid innovation mean/std/support, NEVER GT or teacher.

    Innovations are measured at donor anchors against the donor base estimate,
    not invented by comparing each empty target to a far-away sparse depth.
    Mixed surfaces have larger dispersion, exposed to the head rather than
    turning the pooled mean into an intermediate hard anchor.
    """
    valid=mask.float()
    error=valid*(sparse.float()-base.float())
    mass=F.avg_pool2d(valid,7,1,3)
    mean=F.avg_pool2d(error,7,1,3)/mass.clamp_min(1e-6)
    second=F.avg_pool2d(error.square(),7,1,3)/mass.clamp_min(1e-6)
    var=(second-mean.square()).clamp_min(0)
    support=(mass>0).float()
    # Smooth at zero variance: finite gradients even for one/zero anchor.
    std=((var+1e-6).sqrt()-.001)*support
    return torch.cat((mean/20,std/20,mass),1)



class InnovationReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.diagnostics=True
        self.body=nn.Sequential(ConvBN(80,24),LiteBlock(24))
        self.blend=nn.Conv2d(24,4,1);self.delta=nn.Conv2d(24,4,1)
        self.candidates=nn.Conv2d(24,20,1)
        self.log_uncertainty_strength=nn.Parameter(torch.full((4,),math.log(math.expm1(.5))))
        nn.init.zeros_(self.delta.weight);nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.blend.weight);nn.init.constant_(self.blend.bias,-1.4)
        nn.init.zeros_(self.candidates.weight);nn.init.zeros_(self.candidates.bias)
        with torch.no_grad():self.candidates.bias[:4].fill_(1.5)
        # Hidden24 + dense P4(48) + packed mean/std/support(12) + U(4).
        self.metric=nn.Sequential(nn.Conv2d(88,32,1),nn.SiLU(),
                                  nn.Conv2d(32,32,3,padding=2,dilation=2,groups=32),nn.SiLU(),
                                  nn.Conv2d(32,4,1))
        nn.init.zeros_(self.metric[-1].weight); nn.init.zeros_(self.metric[-1].bias)

    def forward(self,j,context,g4,base,sparse,mask,barrier,p4):
        sensor=F.pixel_unshuffle(torch.cat((base/120,mask,mask*(sparse-base)/20,sparse/120),1),2)
        hidden=self.body(torch.cat((context,g4,sensor),1))
        inv,valid=candidate_queries(j)
        bphase=F.pixel_unshuffle(base.float(),2)[:,None]
        penalty=((inv.reciprocal()-bphase).abs()/(1+.1*bphase)).clamp_max(6)
        b=barrier.float().sigmoid(); east,south=b[:,:1],b[:,1:2]
        west=F.pad(east[...,:-1],(1,0,0,0)); north=F.pad(south[...,:-1,:],(0,0,1,0))
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
        # Enhance the EXISTING metric head; do not stack a second larger depth bound.
        raw=self.delta(hidden).float()+correction
        delta=(1+.05*base)*F.pixel_shuffle(raw.tanh(),2)
        final=(base+geometric+delta).clamp(.1,120)
        diagnostics={}
        if self.diagnostics:
            diagnostics={"query_uncertainty":F.pixel_shuffle(uncertainty,2),
                     "query_uncertainty_mean":uncertainty.mean(),"query_gate_mean":gate.mean(),
                     "query_center_weight_mean":weights[:,0].mean(),
                     "query_entropy_mean":-(weights*weights.clamp_min(1e-8).log()).sum(1).mean(),
                     "innovation_abs_mean_m":(innovation[:,:1]*20).abs().mean(),
                     "innovation_support_fraction":(innovation[:,2:3]>0).float().mean(),
                     "innovation_head_raw_abs":correction.abs().mean(),
                     "metric_head_saturation_fraction":(raw.tanh().abs()>.9).float().mean()}
        return final,query,geometric,delta,diagnostics


class AnchorFlowEdge(nn.Module):
    def __init__(self, pretrained=False, flow_steps=2,
                 encoder="mobilenetv4_conv_small_050.e3000_r224_in1k", model_name="v10_lite"):
        super().__init__()
        if model_name != "v10_lite":
            raise ValueError(f"Unknown V8 model: {model_name}")
        import timm
        self.encoder = timm.create_model(encoder,pretrained=pretrained,features_only=True,out_indices=(1,2,3,4))
        info = self.encoder.feature_info
        if tuple(info.reduction())!=(4,8,16,32):
            raise ValueError("Expected F4/F8/F16/F32")
        self.register_buffer("rgb_mean",torch.tensor(self.encoder.pretrained_cfg["mean"]).view(1,3,1,1))
        self.register_buffer("rgb_std",torch.tensor(self.encoder.pretrained_cfg["std"]).view(1,3,1,1))
        self.sparse = SparsePyramid()
        self.context32 = Context32(info.channels()[-1])
        self.decoder = PyramidDecoder(info.channels()[:3])
        self.guidance = nn.Sequential(ConvBN(12,8),ConvBN(8,8,3,8))
        self.dynamics = FixedJetDynamics(flow_steps,feedback=True)
        self.up4_2 = PhaseUpsample(83,24,1.,.05)
        self.phase2 = InnovationReadout()
        self.context2 = ConvBN(48,8)
        self.up2_1 = PhaseUpsample(19,16,.5,.02)
        self.detail1 = PhaseDetailAndTrust()

    @staticmethod
    def sensor_state(depth,sparse,mask,scale):
        density=F.avg_pool2d(mask,scale,scale)
        mean=F.avg_pool2d(sparse*mask,scale,scale)/density.clamp_min(1e-6)
        valid=(density>0).float()
        return torch.cat((depth/120,valid,valid*(mean-depth)/120),1)

    def forward(self,rgb,sparse,mask,K):
        valid=(mask>.5)&(sparse>.1)&(sparse<120)&torch.isfinite(sparse)
        sparse=torch.where(valid,sparse.float(),torch.zeros_like(sparse,dtype=torch.float32)); mask=valid.float()
        f4,f8,f16,f32=self.encoder((rgb-self.rgb_mean)/self.rgb_std)
        f16=self.context32(f32,f16)
        sparse_features,states=self.sparse(sparse,mask,K.float())
        p4,d16,d8,z0=self.decoder((f4,f8,f16),sparse_features,states[0]); d0=z0.exp()
        g2=self.guidance(F.pixel_unshuffle(rgb,2)); g4=F.pixel_unshuffle(g2,2)
        j,context,diagnostics=self.dynamics(p4,g4,d0,states[0]); d4=j[:,:1].reciprocal()
        d2_base=self.up4_2(d4,torch.cat((p4,g4,self.sensor_state(d4,sparse,mask,4)),1))
        sensor2=self.sensor_state(d2_base,sparse,mask,2)
        d2,query,geometric,phase_delta,query_stats=self.phase2(j,context,g4,d2_base,
            (sensor2[:,2:3]*120+d2_base)*sensor2[:,1:2],sensor2[:,1:2],diagnostics["surface_barrier_logits"],p4)
        p2=F.interpolate(self.context2(p4),size=g2.shape[-2:],mode="nearest")
        rich=self.phase_context(p4) if self.phase_context_enabled else None
        if rich is not None:rich=F.interpolate(rich,size=g2.shape[-2:],mode="nearest")
        d1_base=self.up2_1(d2,torch.cat((p2,g2,self.sensor_state(d2,sparse,mask,2)),1),rich)
        d1,final,gate,logits,delta1=self.detail1(rgb,sparse,mask,d1_base,p2,g2,rich)
        return {"D16":d16,"D8":d8,"D0":d0,"D4":d4,"D2_base":d2_base,"D2_query":query,"D2":d2,
                "D1_base":d1_base,"D1":d1,"D_full":final,"D_hard":torch.where(valid,sparse,d1),
                "sensor_gate":gate,"sensor_logits":logits,"delta1":delta1,"phase2_delta":phase_delta,
                "jet_phase_delta":geometric,**diagnostics,**query_stats}

    def freeze_encoder_bn(self):
        for module in self.encoder.modules():
            if isinstance(module,nn.BatchNorm2d):
                module.eval()



class Deploy(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model=model
    def forward(self,rgb,sparse,mask,K):
        return self.model(rgb,sparse,mask,K)["D_full"]

