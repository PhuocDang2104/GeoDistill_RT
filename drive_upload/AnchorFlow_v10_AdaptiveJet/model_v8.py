"""AnchorFlow V8: fixed-horizon, feedback-conditioned projective jet dynamics.

Only RGB/sparse/mask/K enter inference. No ODE solver, checkpoint migration,
learned optical displacement, ConvGRU hidden state, or full-resolution feature CNN.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from core import ConvBN, LiteBlock, SparsePyramid, PyramidDecoder, PhaseUpsample
from support import Context32, PhaseDetailAndTrust


def adjacent(x):
    """E/W/S/N neighbours. Boundary rates, not padding, enforce zero flux."""
    return torch.stack((F.pad(x[..., 1:], (0, 1, 0, 0), mode="replicate"),
                        F.pad(x[..., :-1], (1, 0, 0, 0), mode="replicate"),
                        F.pad(x[..., 1:, :], (0, 0, 0, 1), mode="replicate"),
                        F.pad(x[..., :-1, :], (0, 0, 1, 0), mode="replicate")), 1)


def derivatives(x):
    east, west, south, north = adjacent(x).unbind(1)
    dxp, dxm, dyp, dym = east-x, x-west, south-x, x-north
    gx = torch.where(dxp.abs() < dxm.abs(), dxp, dxm)
    gy = torch.where(dyp.abs() < dym.abs(), dyp, dym)
    gx = torch.cat((dxp[..., :1], gx[..., 1:-1], dxm[..., -1:]), -1)
    gy = torch.cat((dyp[..., :1, :], gy[..., 1:-1, :], dym[..., -1:, :]), -2)
    return gx, gy


def translate(j, dx, dy):
    """Exact change of origin for a quadratic jet; offsets in quarter-grid cells."""
    v, gx, gy, hxx, hxy, hyy = j.split(1, dim=-3)
    return torch.cat((v+gx*dx+gy*dy+.5*hxx*dx*dx+hxy*dx*dy+.5*hyy*dy*dy,
                      gx+hxx*dx+hxy*dy, gy+hxy*dx+hyy*dy, hxx, hxy, hyy), dim=-3)


def phase_values(j):
    v, gx, gy, hxx, hxy, hyy = j.split(1, 1)
    return torch.cat([v+gx*x+gy*y+.5*hxx*x*x+hxy*x*y+.5*hyy*y*y
                      for y, x in ((-.25,-.25),(-.25,.25),(.25,-.25),(.25,.25))], 1)


def edge_weights(rates):
    """Two undirected edge rates -> four neighbours, reciprocal/zero-flux."""
    east = F.pad(rates[:, :1, :, :-1], (0, 1, 0, 0))
    west = F.pad(rates[:, :1, :, :-1], (1, 0, 0, 0))
    south = F.pad(rates[:, 1:, :-1, :], (0, 0, 0, 1))
    north = F.pad(rates[:, 1:, :-1, :], (0, 0, 1, 0))
    return torch.stack((east, west, south, north), 1)


class JetDynamics(nn.Module):
    """Recompute both reaction and conductance from j^k with shared weights.

    Projected explicit Euler, dt=1/3, three steps. FP32 analytic state;
    learned convolutions autocast. No BN inside the shared vector field.
    """
    def __init__(self, steps=3, feedback=True):
        super().__init__()
        if steps != 3:
            raise ValueError("Canonical V8 fixes three Euler steps; dt and schema depend on it")
        self.steps, self.feedback = steps, feedback
        self.context = nn.Sequential(ConvBN(87, 32), LiteBlock(32),
                                     nn.Conv2d(32, 32, 5, padding=2, groups=32), nn.SiLU())
        self.barrier = nn.Conv2d(32, 2, 1)
        self.state = nn.Conv2d(14, 32, 1)
        self.update = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1, groups=32), nn.SiLU(),
                                    nn.Conv2d(32, 32, 1), nn.SiLU())
        self.reaction = nn.Conv2d(32, 6, 1)
        self.conductance = nn.Conv2d(32, 2, 1)
        self.sensor = nn.Conv2d(32, 1, 1)
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

    @staticmethod
    def base_jet(depth):
        v = depth.float().clamp(.1, 120).reciprocal()
        # Coarse value remains differentiable; avoid higher-order image derivative backprop.
        gx, gy = derivatives(v.detach())
        hxx, hxy_a = derivatives(gx)
        hxy_b, hyy = derivatives(gy)
        return JetDynamics.project(torch.cat((v,gx,gy,hxx,.5*(hxy_a+hxy_b),hyy), 1))

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
                                torch.full_like(v,tau)), 1)
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
        return forcing, weights

    def forward(self, p4, g4, depth, state):
        mean, valid, density, _, _, spread = state
        sensor_features = torch.cat((depth/120,valid*(mean-depth)/20,valid,density,
                                     spread.clamp_max(2),mean/120,valid*torch.exp(-4*spread)),1)
        context = self.context(torch.cat((p4,g4,sensor_features),1))
        barrier = self.barrier(context).float()
        j0 = self.base_jet(depth)
        j = j0
        dt = 1/self.steps
        diagnostics = {"D4_step0":depth, "surface_barrier_logits":barrier}
        dx,dy = self.dx.view(1,4,1,1,1),self.dy.view(1,4,1,1,1)
        for k in range(self.steps):
            # Optional frozen-feedback control has identical weights/compute/recipe.
            observed = j if self.feedback else j0
            forcing,weights = self.vector_field(context,barrier,observed,j0,state,k*dt)
            transported = translate(adjacent(j),dx,dy)
            transport = (weights*(transported-j[:,None])).sum(1)
            next_j = self.project(j+dt*(forcing+transport))
            next_depth = next_j[:, :1].reciprocal()
            diagnostics.update({f"D4_step{k+1}":next_depth,
                                f"dynamics_state_change_{k+1}":((next_j-j)/j0[:, :1]).abs().mean(),
                                f"dynamics_forcing_{k+1}":(forcing/j0[:, :1]).abs().mean(),
                                f"dynamics_transport_{k+1}":(transport/j0[:, :1]).abs().mean(),
                                f"dynamics_mass_{k+1}":(dt*weights.sum(1)).mean(),
                                f"dynamics_sensor_error_m_{k+1}":((mean-next_depth).abs()*valid).sum()/valid.sum().clamp_min(1)})
            j = next_j
        diagnostics["delta4"] = j[:, :1].reciprocal()-depth
        diagnostics["jet_gradient_abs"] = j[:,1:3].abs().mean()
        diagnostics["jet_hessian_abs"] = j[:,3:].abs().mean()
        return j, context, diagnostics


class JetPhaseReadout(nn.Module):
    """Continuous jet query + cheap learned metric detail at quarter grid."""
    def __init__(self):
        super().__init__()
        # Dynamics context32 + packed learned guidance32 + sensor phases16.
        self.body = nn.Sequential(ConvBN(80,24),LiteBlock(24))
        self.blend = nn.Conv2d(24,4,1)
        self.delta = nn.Conv2d(24,4,1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.blend.weight)
        nn.init.constant_(self.blend.bias,-1.4)

    def forward(self, j, context, g4, base, sparse, mask):
        sensor = F.pixel_unshuffle(torch.cat((base/120,mask,
                                             mask*(sparse-base)/20,sparse/120),1),2)
        hidden = self.body(torch.cat((context,g4,sensor),1))
        query = F.pixel_shuffle(phase_values(j).clamp(1/120,10).reciprocal(),2)
        gate = F.pixel_shuffle(self.blend(hidden).float().sigmoid(),2)
        limit = 1+.1*base
        geometric = gate*(query-base).clamp(-limit,limit)
        delta = (1+.05*base)*F.pixel_shuffle(self.delta(hidden).float().tanh(),2)
        return (base+geometric+delta).clamp(.1,120), query, geometric, delta


class AnchorFlowEdge(nn.Module):
    def __init__(self, pretrained=False, flow_steps=3,
                 encoder="mobilenetv4_conv_small_050.e3000_r224_in1k", model_name="v8_dynamics"):
        super().__init__()
        if model_name not in ("v8_dynamics","v8_frozen_feedback"):
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
        self.dynamics = JetDynamics(flow_steps,feedback=model_name=="v8_dynamics")
        self.up4_2 = PhaseUpsample(83,24,1.,.05)
        self.phase2 = JetPhaseReadout()
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
        sparse=torch.where(valid,sparse.float(),torch.zeros_like(sparse,dtype=torch.float32))
        mask=valid.float()
        f4,f8,f16,f32=self.encoder((rgb-self.rgb_mean)/self.rgb_std)
        f16=self.context32(f32,f16)
        sparse_features,states=self.sparse(sparse,mask,K.float())
        p4,d16,d8,z0=self.decoder((f4,f8,f16),sparse_features,states[0])
        d0=z0.exp()
        g2=self.guidance(F.pixel_unshuffle(rgb,2))
        g4=F.pixel_unshuffle(g2,2)
        j,context,diagnostics=self.dynamics(p4,g4,d0,states[0])
        d4=j[:, :1].reciprocal()
        d2_base=self.up4_2(d4,torch.cat((p4,g4,self.sensor_state(d4,sparse,mask,4)),1))
        sensor2=self.sensor_state(d2_base,sparse,mask,2)
        d2,query,geometric,phase_delta=self.phase2(j,context,g4,d2_base,(sensor2[:,2:3]*120+d2_base)*sensor2[:,1:2],
                                                sensor2[:,1:2])
        p2=F.interpolate(self.context2(p4),size=g2.shape[-2:],mode="nearest")
        d1_base=self.up2_1(d2,torch.cat((p2,g2,self.sensor_state(d2,sparse,mask,2)),1))
        d1,final,gate,logits,delta1=self.detail1(rgb,sparse,mask,d1_base,p2,g2)
        return {"D16":d16,"D8":d8,"D0":d0,"D4":d4,"D2_base":d2_base,"D2_query":query,"D2":d2,
                "D1_base":d1_base,"D1":d1,"D_full":final,"D_hard":torch.where(valid,sparse,d1),
                "sensor_gate":gate,"sensor_logits":logits,"delta1":delta1,
                "phase2_delta":phase_delta,"jet_phase_delta":geometric,**diagnostics}

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
