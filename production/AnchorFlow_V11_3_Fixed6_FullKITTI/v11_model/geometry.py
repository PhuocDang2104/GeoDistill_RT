"""A genuine neural IVP in a bounded geometric chart, no projected Euler.

The network defines a state/time-dependent derivative. A non-learned RK solver
integrates chart coordinates. Smooth decoding, NOT solver-step projection,
keeps physical inverse depth, slopes and Hessians inside the readout domain.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from core import ConvBN, LiteBlock
from geometry_primitives import adjacent, translate, edge_weights, limited_derivatives, candidate_queries
from ode_solver import solve

LOG_MIN = math.log(1/120)
LOG_SPAN = math.log(1200.)


def encode_jet(j):
    """Interior chart initialization; epsilon only affects the initial mapping."""
    v = j[:,:1].float().clamp(1/120,10.)
    q = ((v.log()-LOG_MIN)/LOG_SPAN).clamp(1e-6,1-1e-6)
    z0 = q.log() - torch.log1p(-q)
    scales = j.new_tensor([.5,.5,.25,.25,.25]).view(1,5,1,1)
    ratios = (j[:,1:]/(scales*v)).clamp(-.9999,.9999)
    # Exact atanh identity using portable ONNX17 Log/Add/Sub primitives.
    derivative_chart=.5*(torch.log1p(ratios)-torch.log1p(-ratios))
    return torch.cat((z0,derivative_chart),1)


def decode_jet(z):
    """A smooth bounded map for every finite chart coordinate. No hard clip."""
    v = (LOG_MIN + LOG_SPAN*z[:,:1].sigmoid()).exp()
    scales = z.new_tensor([.5,.5,.25,.25,.25]).view(1,5,1,1)
    return torch.cat((v,scales*v*z[:,1:].tanh()),1)


class JetNODE(nn.Module):
    def __init__(self, steps=2, feedback=True):
        super().__init__()
        if steps != 2 or not feedback:
            raise ValueError('V11 has two observation times; state feedback is mandatory')
        self.diagnostics = True
        self.method, self.rtol, self.atol = 'bosh3', .01, .001
        self.first_step, self.max_step = .25, .25
        self.force_observation_steps = False
        self.max_nfe, self.max_num_steps, self.fixed_steps = 193, 64, 4
        self.last_solver_report = {}
        self.context=nn.Sequential(ConvBN(87,32),LiteBlock(32),
                                   nn.Conv2d(32,32,5,padding=2,groups=32),nn.SiLU())
        self.barrier=nn.Conv2d(32,2,1)
        self.state=nn.Conv2d(14,32,1)
        self.update=nn.Sequential(nn.Conv2d(32,32,3,padding=1,groups=32),nn.SiLU(),
                                  nn.Conv2d(32,32,1),nn.SiLU())
        self.reaction=nn.Conv2d(32,6,1)
        self.conductance=nn.Conv2d(32,2,1)
        self.sensor=nn.Conv2d(32,1,1)
        nn.init.normal_(self.reaction.weight,std=1e-3);nn.init.zeros_(self.reaction.bias)
        nn.init.normal_(self.conductance.weight,std=1e-3);nn.init.constant_(self.conductance.bias,-1.4)
        nn.init.normal_(self.sensor.weight,std=1e-3);nn.init.zeros_(self.sensor.bias)
        nn.init.zeros_(self.barrier.weight);nn.init.constant_(self.barrier.bias,-2.)
        self.register_buffer('dx',torch.tensor([-1.,1.,0.,0.]),persistent=False)
        self.register_buffer('dy',torch.tensor([0.,0.,-1.,1.]),persistent=False)
        self.register_buffer('forcing_limits',torch.tensor([.9,.15,.15,.06,.06,.06]))

    def configure(self,cfg):
        if cfg['ode_terminal_time']!=1. or cfg['ode_field_precision']!='fp32':
            raise ValueError('V11 requires fixed T=1 and FP32 RHS')
        for attr,key in [('method','ode_method'),('rtol','ode_rtol'),('atol','ode_atol'),
                         ('first_step','ode_first_step'),('max_step','ode_max_step'),
                         ('max_nfe','ode_max_nfe'),('max_num_steps','ode_max_num_steps'),
                         ('fixed_steps','ode_fixed_steps')]:setattr(self,attr,cfg[key])
        self.force_observation_steps=bool(cfg.get('ode_force_observation_steps',False))

    def initial_jet(self,depth):
        v=depth.float().clamp(.1,120).reciprocal()
        gx,gy=limited_derivatives(v.detach())
        hxx,hxy_a=limited_derivatives(gx);hxy_b,hyy=limited_derivatives(gy)
        return torch.cat((v,gx,gy,hxx,.5*(hxy_a+hxy_b),hyy),1)

    def vector_field(self,t,z,context,barrier,j0,state):
        # FP32 RHS is intentional: BF16 quantization noise can trigger adaptive
        # step rejection and invalidate a tight numerical-error comparison.
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
            transport=(weights*(translate(adjacent(j),self.dx.view(1,4,1,1,1),
                           self.dy.view(1,4,1,1,1))-j[:,None])).sum(1)
            scales=z.new_tensor([1.,.5,.5,.25,.25,.25]).view(1,6,1,1)
            # Define the ODE IN chart space. This is not a claim that it is the
            # exact Jacobian pullback of the old projected physical-jet field.
            normalized=(forcing+transport)/(v*scales)
            return 2*torch.tanh(normalized/2)

    def forward(self,p4,g4,depth,state):
        mean,valid,density,_,_,spread=state
        sensor_features=torch.cat((depth/120,valid*(mean-depth)/20,valid,density,
                                   spread.clamp_max(2),mean/120,valid*torch.exp(-4*spread)),1)
        context=self.context(torch.cat((p4,g4,sensor_features),1))
        barrier=self.barrier(context).float()
        with torch.autocast(depth.device.type,enabled=False):
            z0=encode_jet(self.initial_jet(depth));j0=decode_jet(z0)
            def field(t,z):return self.vector_field(t,z,context,barrier,j0,state)
            trajectory,report=solve(field,z0,method=self.method,rtol=self.rtol,atol=self.atol,
                first_step=self.first_step,max_step=self.max_step,max_nfe=self.max_nfe,
                max_num_steps=self.max_num_steps,fixed_steps=self.fixed_steps,
                force_observation_steps=self.force_observation_steps)
            half,final=decode_jet(trajectory[0]),decode_jet(trajectory[1])
        self.last_solver_report=report
        diagnostics={'surface_barrier_logits':barrier}
        if self.diagnostics:
            diagnostics.update({'D4_step0':j0[:,:1].reciprocal(),'D4_step1':half[:,:1].reciprocal(),
                'D4_step2':final[:,:1].reciprocal(),
                'dynamics_state_change_1':((half-j0)/j0[:,:1]).abs().mean(),
                'dynamics_state_change_2':((final-half)/j0[:,:1]).abs().mean(),
                'dynamics_nfe':depth.new_tensor(float(report['nfe'])),
                'dynamics_accepted_steps':depth.new_tensor(float(report['accepted_steps'])),
                'dynamics_rejected_steps':depth.new_tensor(float(report['rejected_steps'])),
                'dynamics_terminal_time_mean':depth.new_tensor(report['terminal_time']),
                'delta4':final[:,:1].reciprocal()-depth,
                'jet_gradient_abs':final[:,1:3].abs().mean(),'jet_hessian_abs':final[:,3:].abs().mean()})
        return final,context,diagnostics
