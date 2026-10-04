"""Fixed two-step geometric dynamics + zero-init phase-context bypass.

No controller, stopping branch, teacher input or learned full-resolution feature.
Existing metric residual limits and soft sensor fusion are unchanged.
"""
import torch
from torch import nn
from torch.nn import functional as F
from core import ConvBN,PhaseUpsample
from support import PhaseDetailAndTrust
from model_base import AnchorFlowEdge as Base, Deploy


class ContextPhaseDetail(PhaseDetailAndTrust):
    def forward(self,rgb,sparse,mask,depth,p2,g2,rich=None):
        packed=F.pixel_unshuffle(torch.cat((sparse/120,mask,mask*(sparse-depth)/20,depth/120),1),2)
        hidden=self.body(torch.cat((p2,g2,F.pixel_unshuffle(rgb,2),packed),1))
        if rich is not None:hidden=hidden+rich
        raw=F.pixel_shuffle(self.delta(hidden).float(),2)
        delta=(.5+.05*depth)*raw.tanh()
        refined=(depth+delta).clamp(.1,120)
        logits=F.pixel_shuffle(self.trust_logits(hidden).float(),2)
        tolerance=(.5+.02*refined)*self.log_tolerance.clamp(-1,2).exp()
        confidence_logits=logits-torch.log1p(((sparse-refined)/tolerance).square())
        gate=mask*confidence_logits.sigmoid()
        return refined,(1-gate)*refined+gate*sparse,gate,confidence_logits,delta


class ContextPhaseUpsample(PhaseUpsample):
    def forward(self,depth,features,rich=None):
        hidden=self.trunk(features)
        if rich is not None:hidden=hidden+rich[:,:16]
        # Fixed metric-depth hypotheses; unchanged 3x3 convex phase sampler.
        with torch.autocast(depth.device.type,enabled=False):
            neighbours=F.conv2d(F.pad(depth.float(),(1,1,1,1),mode='replicate'),self.neighbours.float())
        weights=self.weights(hidden).float().reshape(depth.shape[0],4,9,*depth.shape[-2:]).softmax(2)
        value=(weights*neighbours[:,None]).sum(2)
        phases=value+(self.residual_base+self.residual_ratio*value)*self.residual(hidden).float().tanh()
        return F.pixel_shuffle(phases,2).clamp(.1,120)


class AnchorFlowEdge(Base):
    def __init__(self,pretrained=False,flow_steps=2,
                 encoder='mobilenetv4_conv_small_050.e3000_r224_in1k',model_name='v10_lite',
                 phase_context_enabled=True):
        super().__init__(pretrained,flow_steps,encoder,model_name)
        self.phase_context_enabled=phase_context_enabled
        # Shared quarter-grid projection conditions BOTH final phase sampling
        # and detail/sensor readout. No new full-resolution feature processing.
        self.phase_context=nn.Sequential(ConvBN(48,12),ConvBN(12,12,3,12),nn.Conv2d(12,24,1))
        nn.init.zeros_(self.phase_context[-1].weight)
        nn.init.zeros_(self.phase_context[-1].bias)
        self.up2_1=ContextPhaseUpsample(19,16,.5,.02)
        self.detail1=ContextPhaseDetail()

    def set_diagnostics(self,enabled):
        # Telemetry never changes the predicted depth. Deploy skips reductions.
        self.dynamics.diagnostics=bool(enabled)
        self.phase2.diagnostics=bool(enabled)

