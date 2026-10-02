"""Ray-constrained piecewise surface correction, ONLY on the quarter grid.

No lateral point motion / splatting / grid_sample / GT input. Directional barriers
live on undirected edges, not pixels. Fixed two-step execution; scalar geometry
is FP32 under AMP. The signed bounded readout makes migration an EXACT no-op.
"""
import torch
from torch import nn
from torch.nn import functional as F
from core import ConvBN, LiteBlock
from model_v3 import AnchorFlowEdge as V3


def neighbours(t):
    """E, W, S, N; replicated values are masked by edge validity in transport."""
    return torch.cat((F.pad(t[..., 1:], (0,1,0,0), mode="replicate"),
                      F.pad(t[..., :-1], (1,0,0,0), mode="replicate"),
                      F.pad(t[..., 1:, :], (0,0,0,1), mode="replicate"),
                      F.pad(t[..., :-1, :], (0,0,1,0), mode="replicate")), 1)


def directional_weights(logits, barriers):
    """Same edge conductance in either direction; <=1/4 per neighbour.

Self weight is 1-sum(w). Do NOT renormalize neighbour-only weights: reducing
all four conductances must reduce transport, not cancel the barrier.
"""
    horizontal, vertical = logits.float().split(1,1)
    if barriers is not None:
        bh, bv = barriers.float().split(1,1)
        horizontal = horizontal.sigmoid() * (1-bh.sigmoid())
        vertical = vertical.sigmoid() * (1-bv.sigmoid())
    else:
        horizontal, vertical = horizontal.sigmoid(), vertical.sigmoid()
    horizontal = F.pad(horizontal[..., :-1], (0,1,0,0))
    vertical = F.pad(vertical[..., :-1, :], (0,0,0,1))
    return .25 * torch.cat((horizontal, F.pad(horizontal[..., :-1], (1,0,0,0)),
                            vertical, F.pad(vertical[..., :-1, :], (0,0,1,0))), 1)


def plane_offsets(slopes,K):
    sx,sy = slopes.float().split(1,1)
    dx = 4.0/K[:,0,0,None,None,None].float().clamp_min(1)
    dy = 4.0/K[:,1,1,None,None,None].float().clamp_min(1)
    direction_x = torch.cat((-dx,dx,dx*0,dx*0),1)
    direction_y = torch.cat((dy*0,dy*0,-dy,dy),1)
    return neighbours(sx)*direction_x+neighbours(sy)*direction_y


def plane_transport(inverse, slopes, K, weights, offsets=None):
    """For a plane n.X=c, inverse depth is affine in normalized camera x/y.

Transport each neighbour's local inverse-depth plane onto the SAME target ray:
xi(q->p)=xi(q)+a(q)*(x_p-x_q)+b(q)*(y_p-y_q).
Intrinsics correspond to the full image; quarter-grid spacing is four pixels.
"""
    xi = inverse.float()
    if offsets is None:
        offsets = plane_offsets(slopes,K)
    transported = (neighbours(xi)+offsets).clamp(1/120,10)
    return ((1-weights.sum(1,keepdim=True))*xi + (weights*transported).sum(1,keepdim=True)).clamp(1/120,10)


class PiecewiseSurface(nn.Module):
    def __init__(self, barrier_enabled=True):
        super().__init__()
        self.barrier_enabled = barrier_enabled
        # P4(48) + RGB4(3) + depth, innovation, valid, density, spread, x,y(7).
        self.body = nn.Sequential(ConvBN(58,32), LiteBlock(32),
                                  nn.Conv2d(32,32,5,padding=2,groups=32), nn.SiLU())
        self.slopes = nn.Conv2d(32,2,1)
        self.conductance = nn.Conv2d(32,2,1)
        self.barriers = nn.Conv2d(32,2,1)
        self.reaction = nn.Conv2d(32,1,1)
        self.amplitude = nn.Parameter(torch.zeros(()))
        for head in (self.slopes,self.conductance,self.barriers,self.reaction):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.constant_(self.barriers.bias,-2.)

    def forward(self, feature, depth, state, rgb, K):
        mean,valid,density,_,_,spread = state
        h,w = depth.shape[-2:]
        u = (torch.arange(w,device=depth.device,dtype=torch.float32)+.5)*4-.5
        v = (torch.arange(h,device=depth.device,dtype=torch.float32)+.5)*4-.5
        x = (u.view(1,1,1,w)-K[:,0,2,None,None,None])/K[:,0,0,None,None,None].clamp_min(1)
        y = (v.view(1,1,h,1)-K[:,1,2,None,None,None])/K[:,1,1,None,None,None].clamp_min(1)
        sensor = torch.cat((depth/120, (valid*(mean-depth)/20).clamp(-6,6), valid,
                            density,spread.clamp_max(2),x.expand_as(depth),y.expand_as(depth)),1)
        hidden = self.body(torch.cat((feature,F.avg_pool2d(rgb,4,4),sensor),1))
        slopes = .5*self.slopes(hidden).float().tanh()
        affinity = self.conductance(hidden).float()
        barrier = self.barriers(hidden).float()
        weights = directional_weights(affinity,barrier if self.barrier_enabled else None)
        offsets = plane_offsets(slopes,K)  # cache geometry across both steps; no repeated slope shifts
        reaction = self.reaction(hidden).float().tanh()
        amplitude = .5*self.amplitude.tanh()  # signed residual readout, NOT a visibility probability
        original = depth.float()
        current = original
        transport_total = torch.zeros_like(current)
        reaction_total = torch.zeros_like(current)
        for _ in range(2):
            proposal = plane_transport(current.reciprocal(),slopes,K,weights,offsets).reciprocal()
            # Limit each metric step; two explicit steps, no adaptive solver.
            limit = .5+.05*current
            innovation = (proposal-current).clamp(-limit,limit)
            transport_step,reaction_step = .5*amplitude*innovation,.5*limit*reaction
            transport_total = transport_total+transport_step
            reaction_total = reaction_total+reaction_step
            current = (current+transport_step+reaction_step).clamp(.1,120)
        return current,{"D4_base":original,"surface_barrier_logits":barrier,
                        "surface_slopes":slopes,"surface_weights":weights,
                        "surface_delta":current-original,"surface_amplitude":amplitude.reshape(1),
                        "surface_transport_delta":transport_total,"surface_reaction_delta":reaction_total}


class AnchorFlowEdge(V3):
    def __init__(self, pretrained=False, flow_steps=3,
                 encoder="mobilenetv4_conv_small_050.e3000_r224_in1k", barrier_enabled=True):
        super().__init__(pretrained=pretrained,flow_steps=flow_steps,encoder=encoder)
        self.surface = PiecewiseSurface(barrier_enabled)

    def refine_surface(self,p4,d4,state,rgb,K):
        return self.surface(p4,d4,state,rgb,K)
