"""Half-grid, state-dependent jet dynamics: ONE fixed midpoint step, exactly 2 NFE.

No torchdiffeq/error control/rejection/learned timestep. The geometric state is
decoded from the same bounded chart as coarse V11 at both RHS evaluations.
Context is computed once; the tiny shared RHS is FP32 and has no BatchNorm.
"""
import torch
from torch import nn
from torch.nn import functional as F
from geometry import encode_jet, decode_jet
from geometry_primitives import adjacent, translate, edge_weights, limited_derivatives


def fixed_midpoint(field, initial):
    """Explicit midpoint RK2 over local pseudo-time [0,1], exactly two calls."""
    if initial.dtype != torch.float32:
        raise ValueError('Fine NODE state and RK2 must be FP32')
    k1 = field(initial.new_zeros(()), initial)
    middle = initial + .5*k1
    k2 = field(initial.new_tensor(.5), middle)
    return initial + k2, middle


class FineJetNODE(nn.Module):
    """981 parameters; normalized reaction + sensor source + local jet transport."""
    def __init__(self):
        super().__init__()
        self.diagnostics = True
        self.last_solver_report = {}
        # Existing p2(8) + learned RGB g2(8) + sensor/depth context(7).
        self.context = nn.Sequential(nn.Conv2d(23,12,1), nn.SiLU(),
                                     nn.Conv2d(12,12,3,padding=1,groups=12), nn.SiLU())
        self.state = nn.Conv2d(14,12,1)
        self.update = nn.Sequential(nn.Conv2d(12,12,3,padding=1,groups=12), nn.SiLU(),
                                    nn.Conv2d(12,12,1), nn.SiLU())
        self.reaction = nn.Conv2d(12,6,1)
        self.sensor = nn.Conv2d(12,1,1)
        self.conductance = nn.Conv2d(12,2,1)
        # Near-identity but not an artificially zero-gradient branch. Small
        # initial source/transport gates are learned from epoch0, not scheduled.
        nn.init.normal_(self.reaction.weight,std=1e-3); nn.init.zeros_(self.reaction.bias)
        nn.init.normal_(self.sensor.weight,std=1e-3); nn.init.constant_(self.sensor.bias,-4.)
        nn.init.normal_(self.conductance.weight,std=1e-3); nn.init.constant_(self.conductance.bias,-4.)
        self.register_buffer('forcing_limits',torch.tensor([.08,.06,.06,.03,.03,.03]))
        self.register_buffer('chart_scales',torch.tensor([1.,.5,.5,.25,.25,.25]),persistent=False)
        self.register_buffer('dx',torch.tensor([-1.,1.,0.,0.]),persistent=False)
        self.register_buffer('dy',torch.tensor([0.,0.,-1.,1.]),persistent=False)

    @staticmethod
    def sensor_summary(sparse,mask):
        """Valid-area mean and spread at 1/2; never GT/teacher or hard anchoring."""
        density = F.avg_pool2d(mask.float(),2,2)
        mean = F.avg_pool2d(sparse.float()*mask,2,2)/density.clamp_min(1e-6)
        second = F.avg_pool2d(sparse.float().square()*mask,2,2)/density.clamp_min(1e-6)
        valid = (density>0).float()
        var = (second-mean.square()).clamp_min(0)
        std = ((var+1e-6).sqrt()-.001)*valid
        spread = std/(.5+.02*mean)
        return mean,valid,density,spread

    @staticmethod
    def initial_jet(depth):
        v = depth.float().reciprocal()
        gx,gy = limited_derivatives(v.detach())
        hxx,hxy_a = limited_derivatives(gx)
        hxy_b,hyy = limited_derivatives(gy)
        return torch.cat((v,gx,gy,hxx,.5*(hxy_a+hxy_b),hyy),1)

    def vector_field(self,t,z,context,j0,sensor,barrier):
        with torch.autocast(z.device.type,enabled=False):
            j = decode_jet(z.float()); v = j[:,:1]; v0 = j0[:,:1]
            depth = v.reciprocal(); mean,valid,density,spread = sensor
            reliability = valid*torch.exp(-4*spread)
            inverse_error = valid*(mean.clamp_min(.1).reciprocal()-v)/v
            metric_error = valid*(mean-depth)/20
            tau = t.float().reshape(1,1,1,1).expand_as(v)
            features = torch.cat((j/v0,v/v0-1,metric_error.clamp(-4,4),
                inverse_error.clamp(-2,2),valid,density,spread.clamp_max(2),reliability,tau),1)
            hidden = self.update(context.float()+self.state(features))
            reaction = self.reaction(hidden).tanh()*v*self.forcing_limits.view(1,6,1,1)
            source = .1*self.sensor(hidden).sigmoid()*reliability*inverse_error.clamp(-1,1)*v
            reaction = reaction+torch.cat((source,torch.zeros_like(j[:,1:])),1)
            # Reuse coarse learned barrier; fine conductance can also learn RGB
            # discontinuities from g2. Sensor support is not an intermediate anchor.
            rates = .15*self.conductance(hidden).sigmoid()*(1-barrier.sigmoid())
            neighbour_v = adjacent(v)
            neighbour_depth = neighbour_v.reciprocal()
            compatibility = torch.exp(-(neighbour_depth-depth[:,None]).abs()/
                                       (.3+.01*(neighbour_depth+depth[:,None])))
            weights = edge_weights(rates)*(.05+.95*compatibility)
            transported = translate(adjacent(j),self.dx.view(1,4,1,1,1),self.dy.view(1,4,1,1,1))
            diffusion = (weights*(transported-j[:,None])).sum(1)
            rhs = (reaction+diffusion)/(v*self.chart_scales.view(1,6,1,1))
            return 2*torch.tanh(rhs/2)

    def forward(self,p2,g2,depth,sparse,mask,coarse_barrier):
        with torch.autocast(depth.device.type,enabled=False):
            sensor = self.sensor_summary(sparse,mask)
            mean,valid,density,spread = sensor
            reliability = valid*torch.exp(-4*spread)
            sensor_features = torch.cat((depth.float()/120,mean/120,valid,density,
                spread.clamp_max(2),reliability,valid*(mean-depth.float())/20),1)
            z0 = encode_jet(self.initial_jet(depth)); j0 = decode_jet(z0)
            barrier = F.interpolate(coarse_barrier.float(),size=depth.shape[-2:],mode='nearest')
        # BF16 context under outer autocast; once per image. Both RHS calls FP32.
        context = self.context(torch.cat((p2,g2,sensor_features),1))
        with torch.autocast(depth.device.type,enabled=False):
            calls = 0
            def field(t,z):
                nonlocal calls
                calls += 1
                return self.vector_field(t,z,context,j0,sensor,barrier)
            terminal,middle = fixed_midpoint(field,z0)
            final = decode_jet(terminal)
            output = final[:,:1].reciprocal()
        # Plain scalar metadata: no GPU .item(), no adaptive host loop.
        self.last_solver_report = {'method':'fixed_midpoint_rk2','nfe':calls,
            'fixed_steps':1,'step_size':1.,'terminal_time':1.,'rejected_steps':0,
            'tolerance_controlled':False,'state_precision':'fp32','chart_state':True}
        diagnostics = {}
        if self.diagnostics:
            diagnostics = {'fine_node_nfe':output.new_tensor(float(calls)),
                'fine_node_abs_delta2_mean_m':(output-depth).abs().mean(),
                'fine_node_mid_chart_change':(middle-z0).abs().mean(),
                'fine_node_terminal_chart_change':(terminal-z0).abs().mean()}
        return output,diagnostics
