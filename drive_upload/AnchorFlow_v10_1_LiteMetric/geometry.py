from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from core import ConvBN,LiteBlock

def adjacent(x):
    """E/W/S/N neighbours. Boundary rates, not padding, enforce zero flux."""
    return torch.stack((F.pad(x[..., 1:], (0, 1, 0, 0), mode="replicate"),
                        F.pad(x[..., :-1], (1, 0, 0, 0), mode="replicate"),
                        F.pad(x[..., 1:, :], (0, 0, 0, 1), mode="replicate"),
                        F.pad(x[..., :-1, :], (0, 0, 1, 0), mode="replicate")), 1)



def translate(j, dx, dy):
    """Exact change of origin for a quadratic jet; offsets in quarter-grid cells."""
    v, gx, gy, hxx, hxy, hyy = j.split(1, dim=-3)
    return torch.cat((v+gx*dx+gy*dy+.5*hxx*dx*dx+hxy*dx*dy+.5*hyy*dy*dy,
                      gx+hxx*dx+hxy*dy, gy+hxy*dx+hyy*dy, hxx, hxy, hyy), dim=-3)



def edge_weights(rates):
    """Two undirected edge rates -> four neighbours, reciprocal/zero-flux."""
    east = F.pad(rates[:, :1, :, :-1], (0, 1, 0, 0))
    west = F.pad(rates[:, :1, :, :-1], (1, 0, 0, 0))
    south = F.pad(rates[:, 1:, :-1, :], (0, 0, 0, 1))
    north = F.pad(rates[:, 1:, :-1, :], (0, 0, 1, 0))
    return torch.stack((east, west, south, north), 1)

def minmod(a,b):
    # Continuous equivalent of 0.5*(sign(a)+sign(b))*min(abs(a),abs(b)).
    # No tie-sensitive selection of oppositely signed slopes, no Sign operator.
    return F.relu(torch.minimum(a,b))-F.relu(torch.minimum(-a,-b))



def limited_derivatives(x):
    east,west,south,north=adjacent(x).unbind(1)
    a,b,c,d=east-x,x-west,south-x,x-north
    gx,gy=minmod(a,b),minmod(c,d)
    gx=torch.cat((a[...,:1],gx[...,1:-1],b[...,-1:]),-1)
    gy=torch.cat((c[...,:1,:],gy[...,1:-1,:],d[...,-1:,:]),-2)
    return gx,gy

def candidate_queries(j):
    """Return inverse depths B,5,4,H,W and legal-neighbour mask B,5,1,H,W.

    Candidate order center,E,W,S,N. Each jet is evaluated at the target phase
    expressed relative to that candidate's origin. No interpolation/grid_sample.
    """
    jets = torch.cat((j[:, None], adjacent(j)), 1).float()
    ox = j.new_tensor([0, 1, -1, 0, 0]).view(1, 5, 1, 1, 1)
    oy = j.new_tensor([0, 0, 0, 1, -1]).view(1, 5, 1, 1, 1)
    # Query only the value component, all 5 origins x 4 phases in one broadcast.
    # Equivalent to translate(...)[..., :1], without materializing unused
    # transported gradient/Hessian components or four separate six-state cats.
    sx = jets.new_tensor([-.25,.25,-.25,.25]).view(1,1,4,1,1)-ox
    sy = jets.new_tensor([-.25,-.25,.25,.25]).view(1,1,4,1,1)-oy
    v,gx,gy,hxx,hxy,hyy = jets.split(1,dim=2)
    candidates = (v+gx*sx+gy*sy+.5*hxx*sx*sx+hxy*sx*sy+.5*hyy*sy*sy).clamp(1/120,10.)
    one = torch.ones_like(j[:, :1])
    valid = torch.cat((one[:, None],
                      torch.stack((F.pad(one[..., 1:], (0, 1, 0, 0)),
                                   F.pad(one[..., :-1], (1, 0, 0, 0)),
                                   F.pad(one[..., 1:, :], (0, 0, 0, 1)),
                                   F.pad(one[..., :-1, :], (0, 0, 1, 0))), 1)), 1)
    return candidates, valid


class FixedJetDynamics(nn.Module):
    """Recompute both reaction and conductance from j^k with shared weights.

    Projected explicit Euler, learned h in [1/6,1/3], fixed call count. FP32 state;
    learned convolutions autocast. No BN inside the shared vector field.
    """
    def __init__(self, steps=3, feedback=True):
        super().__init__()
        if steps not in (2,3):
            raise ValueError("Fixed graph supports 2 or 3 steps, h=1/3")
        self.steps, self.feedback = steps, feedback
        self.diagnostics = True
        self.learned_step_size=True
        self.step_min,self.step_max=1/6,1/3
        self.context = nn.Sequential(ConvBN(87, 32), LiteBlock(32),
                                     nn.Conv2d(32, 32, 5, padding=2, groups=32), nn.SiLU())
        self.barrier = nn.Conv2d(32, 2, 1)
        self.state = nn.Conv2d(14, 32, 1)
        self.update = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1, groups=32), nn.SiLU(),
                                    nn.Conv2d(32, 32, 1), nn.SiLU())
        self.reaction = nn.Conv2d(32, 6, 1)
        self.conductance = nn.Conv2d(32, 2, 1)
        self.sensor = nn.Conv2d(32, 1, 1)
        # One scalar step per sample, conditioned on the current shared field
        # feature. No stop head, multi-statistic MLP or host-side branching.
        self.step_head=nn.Conv2d(32,1,1)
        nn.init.zeros_(self.step_head.weight)
        nn.init.zeros_(self.step_head.bias) # fresh h=.25, not saturated at max
        nn.init.normal_(self.reaction.weight, std=1e-3)
        nn.init.zeros_(self.reaction.bias)
        nn.init.normal_(self.conductance.weight, std=1e-3)
        nn.init.constant_(self.conductance.bias, -1.4)
        nn.init.normal_(self.sensor.weight, std=1e-3)
        nn.init.zeros_(self.sensor.bias)
        nn.init.zeros_(self.barrier.weight)
        nn.init.constant_(self.barrier.bias, -2.)
        # Rank-1 buffers: safe for .to(memory_format=channels_last).
        self.register_buffer("dx", torch.tensor([-1.,1.,0.,0.]), persistent=False)
        self.register_buffer("dy", torch.tensor([0.,0.,-1.,1.]), persistent=False)
        self.register_buffer("forcing_limits", torch.tensor([.9,.15,.15,.06,.06,.06]))

    def base_jet(self,depth):
        v=depth.float().clamp(.1,120).reciprocal()
        gx,gy=limited_derivatives(v.detach())
        hxx,hxy_a=limited_derivatives(gx)
        hxy_b,hyy=limited_derivatives(gy)
        return self.project(torch.cat((v,gx,gy,hxx,.5*(hxy_a+hxy_b),hyy),1))

    @staticmethod
    def project(j):
        v = j[:, :1].clamp(1/120, 10.)
        g, h = j[:, 1:3], j[:, 3:]
        return torch.cat((v, torch.maximum(torch.minimum(g,.5*v),-.5*v),
                           torch.maximum(torch.minimum(h,.25*v),-.25*v)), 1)

    def vector_field(self, context, barrier, j, j0, state, tau):
        mean, valid, density, _, _, spread = state
        v = j[:, :1]
        depth = v.reciprocal()
        reliability = valid * torch.exp(-4*spread)
        innovation = valid * (mean.clamp_min(.1).reciprocal()-v) / v.clamp_min(1/120)
        metric_error = valid * (mean-depth) / 20
        state_input = torch.cat((j/j0[:, :1].clamp_min(1/120),
                                (v/j0[:, :1].clamp_min(1/120)-1),
                                metric_error.clamp(-4,4), innovation.clamp(-2,2),
                                valid,density,spread.clamp_max(2),reliability,
                                tau[:,None,None,None].expand_as(v)), 1)
        hidden = self.update(context + self.state(state_input))
        forcing = self.reaction(hidden).float().tanh() * v * self.forcing_limits.view(1,6,1,1)
        # Sensor feedback is a soft bounded reaction, never a hard intermediate anchor.
        source = self.sensor(hidden).float().sigmoid() * reliability * innovation.clamp(-1,1) * v
        forcing = forcing + torch.cat((source, torch.zeros_like(j[:, 1:])), 1)
        rates = .6*self.conductance(hidden).float().sigmoid()*(1-barrier.sigmoid())
        # Current-state symmetric compatibility, refreshed each Euler step.
        neighbour_v = adjacent(v)
        mismatch_m = (neighbour_v-v[:,None]).abs() / (neighbour_v*v[:,None]).clamp_min(1e-6)
        tolerance = .5+.015*(depth[:,None]+neighbour_v.clamp_min(1/120).reciprocal())
        compatibility = .05+.95*torch.exp(-mismatch_m/tolerance)
        weights = edge_weights(rates)*compatibility
        if self.learned_step_size:
            logits=self.step_head(hidden).float().flatten(1).mean(1)
            step=self.step_min+(self.step_max-self.step_min)*logits.sigmoid()
        else:step=torch.full_like(tau,self.step_max)
        return forcing, weights, step

    def forward(self, p4, g4, depth, state):
        mean, valid, density, _, _, spread = state
        sensor_features = torch.cat((depth/120,valid*(mean-depth)/20,valid,density,
                                     spread.clamp_max(2),mean/120,valid*torch.exp(-4*spread)),1)
        context = self.context(torch.cat((p4,g4,sensor_features),1))
        barrier = self.barrier(context).float()
        j0 = self.base_jet(depth)
        j = j0
        tau=torch.zeros(depth.shape[0],device=depth.device,dtype=torch.float32)
        diagnostics = {"D4_step0":depth, "surface_barrier_logits":barrier}
        dx,dy = self.dx.view(1,4,1,1,1),self.dy.view(1,4,1,1,1)
        for k in range(self.steps):
            # Optional frozen-feedback control has identical weights/compute/recipe.
            observed = j if self.feedback else j0
            forcing,weights,dt = self.vector_field(context,barrier,observed,j0,state,tau)
            transported = translate(adjacent(j),dx,dy)
            transport = (weights*(transported-j[:,None])).sum(1)
            next_j = self.project(j+dt[:,None,None,None]*(forcing+transport))
            tau=tau+dt
            next_depth = next_j[:, :1].reciprocal()
            if self.diagnostics:
                diagnostics.update({f"D4_step{k+1}":next_depth,
                                    f"dynamics_state_change_{k+1}":((next_j-j)/j0[:, :1]).abs().mean(),
                                    f"dynamics_forcing_{k+1}":(forcing/j0[:, :1]).abs().mean(),
                                    f"dynamics_transport_{k+1}":(transport/j0[:, :1]).abs().mean(),
                                    f"dynamics_mass_{k+1}":(dt[:,None,None,None]*weights.sum(1)).mean(),
                                    f"dynamics_step_mean_{k+1}":dt.mean(),
                                    f"dynamics_step_min_{k+1}":dt.min(),
                                    f"dynamics_step_max_{k+1}":dt.max(),
                                    f"dynamics_sensor_error_m_{k+1}":((mean-next_depth).abs()*valid).sum()/valid.sum().clamp_min(1)})
            j = next_j
        if self.diagnostics:
            diagnostics['dynamics_terminal_time_mean']=tau.mean()
            diagnostics["delta4"] = j[:, :1].reciprocal()-depth
            diagnostics["jet_gradient_abs"] = j[:,1:3].abs().mean()
            diagnostics["jet_hessian_abs"] = j[:,3:].abs().mean()
        return j, context, diagnostics

