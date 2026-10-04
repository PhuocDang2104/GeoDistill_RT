"""Projected neural geometric dynamics with a bounded learned integration budget.

No teacher/GT input. Eager adaptive_batch1 really exits. masked_adaptive executes
four calls but returns the same selected state, never an average of states.
"""
import torch
from torch import nn
from torch.nn import functional as F
from model_baseline import AnchorFlowEdge as Baseline, LimitedJetDynamics
from model_v8 import adjacent,translate,edge_weights,Deploy


def rms(x):
    return ((x.float().square().flatten(1).mean(1)+1e-8).sqrt()-1e-4)


class AdaptiveJetController(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(10,16),nn.SiLU(),nn.Linear(16,16),nn.SiLU(),nn.Linear(16,2))
        nn.init.zeros_(self.net[-1].weight)
        # Fresh training starts near legacy h, conservative four-call policy.
        with torch.no_grad(): self.net[-1].bias.copy_(torch.tensor([4.,-2.]))

    def forward(self,statistics):
        # Tiny controller + reductions remain FP32, also under BF16 CNN autocast.
        with torch.autocast(statistics.device.type,enabled=False):
            return self.net(statistics.float())


class BudgetedJetDynamics(LimitedJetDynamics):
    def __init__(self,limited=True,policy='adaptive',step_min=1/6,step_max=1/3,
                 stop_threshold=.5,exploration=.2):
        super().__init__(3,limited)
        if policy not in ('fixed3','fixed4','learned4','adaptive'): raise ValueError(policy)
        if not 0<step_min<=step_max<=1/3: raise ValueError('Require 0<h_min<=h_max<=1/3')
        if not 0<stop_threshold<1 or not 0<=exploration<=1: raise ValueError('Invalid controller policy')
        self.controller=AdaptiveJetController()
        self.policy,self.step_min,self.step_max=policy,step_min,step_max
        self.stop_threshold,self.exploration=stop_threshold,exploration
        self.mode='masked_adaptive'
        self.measure_branch_sync=False
        self.branch_sync_seconds=0.

    def vector_field(self,context,barrier,j,j0,state,tau):
        """Same bounded V9.1 field, now conditioned on actual per-sample tau."""
        mean,valid,density,_,_,spread=state
        v=j[:,:1]; depth=v.reciprocal()
        reliability=valid*torch.exp(-4*spread)
        innovation=valid*(mean.clamp_min(.1).reciprocal()-v)/v.clamp_min(1/120)
        metric_error=valid*(mean-depth)/20
        state_input=torch.cat((j/j0[:,:1].clamp_min(1/120),
            v/j0[:,:1].clamp_min(1/120)-1,metric_error.clamp(-4,4),innovation.clamp(-2,2),
            valid,density,spread.clamp_max(2),reliability,tau[:,None,None,None].expand_as(v)),1)
        hidden=self.update(context+self.state(state_input))
        forcing=self.reaction(hidden).float().tanh()*v*self.forcing_limits.view(1,6,1,1)
        source=self.sensor(hidden).float().sigmoid()*reliability*innovation.clamp(-1,1)*v
        forcing=forcing+torch.cat((source,torch.zeros_like(j[:,1:])),1)
        rates=.6*self.conductance(hidden).float().sigmoid()*(1-barrier.sigmoid())
        neighbour_v=adjacent(v)
        mismatch=(neighbour_v-v[:,None]).abs()/(neighbour_v*v[:,None]).clamp_min(1e-6)
        tolerance=.5+.015*(depth[:,None]+neighbour_v.clamp_min(1/120).reciprocal())
        weights=edge_weights(rates)*(.05+.95*torch.exp(-mismatch/tolerance))
        return forcing,weights

    @staticmethod
    def statistics(j,previous,j0,state,forcing,transport,weights,barrier,tau):
        mean,valid,*_=state
        v=j[:,:1].clamp_min(1/120)
        error=valid*(mean.clamp_min(.1).reciprocal()-v)/v
        count=valid.flatten(1).sum(1).clamp_min(1)
        innovation_mean=error.abs().flatten(1).sum(1)/count
        innovation_rms=((error.square().flatten(1).sum(1)/count+1e-8).sqrt()-1e-4)
        change=(j-previous)/previous[:,:1].clamp_min(1/120)
        values=(innovation_mean,innovation_rms,valid.flatten(1).mean(1),
            rms(change[:,:1]),rms(change[:,1:]),rms(forcing/v),rms(transport/v),
            weights.flatten(1).mean(1),barrier.sigmoid().flatten(1).mean(1),tau)
        # Stop BCE must not drive geometry toward artificial stop labels.
        return torch.stack(values,1).detach()

    def forward(self,p4,g4,depth,state):
        mean,valid,density,_,_,spread=state
        sensor=torch.cat((depth/120,valid*(mean-depth)/20,valid,density,spread.clamp_max(2),
                          mean/120,valid*torch.exp(-4*spread)),1)
        context=self.context(torch.cat((p4,g4,sensor),1)); barrier=self.barrier(context).float()
        j0=self.base_jet(depth); j=previous=j0
        self.branch_sync_seconds=0.
        tau=torch.zeros(j.shape[0],device=j.device,dtype=torch.float32)
        states,times,hs,stops=[],[],[],[]
        diag={'D4_step0':depth,'surface_barrier_logits':barrier}
        calls=3 if self.policy=='fixed3' else 4
        if self.mode=='adaptive_batch1' and (self.training or j.shape[0]!=1):
            raise ValueError('True conditional execution is eval batch1 only')
        dx,dy=self.dx.view(1,4,1,1,1),self.dy.view(1,4,1,1,1)
        executed=0
        for k in range(calls):
            forcing,weights=self.vector_field(context,barrier,j,j0,state,tau)
            transport=(weights*(translate(adjacent(j),dx,dy)-j[:,None])).sum(1)
            stats=self.statistics(j,previous,j0,state,forcing,transport,weights,barrier,tau)
            if self.policy in ('fixed3','fixed4'):
                h=torch.full_like(tau,1/3)
            else:
                h=self.step_min+(self.step_max-self.step_min)*self.controller(stats)[:,0].sigmoid()
            next_j=self.project(j+h[:,None,None,None]*(forcing+transport))
            next_tau=tau+h
            # Post-update evidence; REUSE this call's forcing/transport diagnostics.
            # No look-ahead state, no fifth field call, no GT inference signal.
            stop_stats=self.statistics(next_j,j,j0,state,forcing,transport,weights,barrier,next_tau)
            stop=self.controller(stop_stats)[:,1] if self.policy=='adaptive' else torch.full_like(tau,-20.)
            states.append(next_j); times.append(next_tau); hs.append(h); stops.append(stop)
            diag.update({f'D4_step{k+1}':next_j[:,:1].reciprocal(),
                f'dynamics_state_change_{k+1}':((next_j-j)/j0[:,:1]).abs().mean(),
                f'dynamics_forcing_{k+1}':(forcing/j0[:,:1]).abs().mean(),
                f'dynamics_transport_{k+1}':(transport/j0[:,:1]).abs().mean(),
                f'dynamics_mass_{k+1}':(h[:,None,None,None]*weights.sum(1)).mean(),
                f'dynamics_sensor_error_m_{k+1}':((mean-next_j[:,:1].reciprocal()).abs()*valid).sum()/valid.sum().clamp_min(1)})
            previous,j,tau=j,next_j,next_tau; executed+=1
            if self.mode=='adaptive_batch1' and self.policy=='adaptive' and k in (1,2):
                # This host scalar synchronizes CUDA: included in wall-clock profile.
                if self.measure_branch_sync:
                    import time
                    started=time.perf_counter()
                    should_stop=bool(stop.sigmoid()[0]>=self.stop_threshold)
                    self.branch_sync_seconds+=time.perf_counter()-started
                else: should_stop=bool(stop.sigmoid()[0]>=self.stop_threshold)
                if should_stop: break
        n=torch.full_like(tau,len(states),dtype=torch.int64)
        if self.policy=='adaptive' and self.mode not in ('static4','adaptive_batch1'):
            n=torch.where(stops[1].sigmoid().detach()>=self.stop_threshold,2,
                          torch.where(stops[2].sigmoid().detach()>=self.stop_threshold,3,4))
            if self.mode in ('force2','force3'): n=torch.full_like(n,int(self.mode[-1]))
            if self.training and self.exploration:
                explore=torch.rand_like(tau)<self.exploration
                n=torch.where(explore,torch.randint(2,5,n.shape,device=n.device),n)
            # Select a realizable geometric state, not a convex blend of trajectories.
            j=torch.where((n==2)[:,None,None,None],states[1],
                          torch.where((n==3)[:,None,None,None],states[2],states[3]))
            tau=torch.where(n==2,times[1],torch.where(n==3,times[2],times[3]))
        # Fixed schema for batching diagnostics; inactive slots have zero h.
        for k in range(len(states),4):
            diag[f'D4_step{k+1}']=states[-1][:,:1].reciprocal()
            hs.append(torch.zeros_like(tau)); stops.append(torch.full_like(tau,20.))
        diag.update(delta4=j[:,:1].reciprocal()-depth,jet_gradient_abs=j[:,1:3].abs().mean(),
                    jet_hessian_abs=j[:,3:].abs().mean(),
                    adaptive_h=torch.stack(hs,1),adaptive_stop_logits=torch.stack(stops,1),
                    adaptive_selected_nfe=n,adaptive_executed_nfe=torch.full_like(n,executed),
                    adaptive_terminal_time=tau)
        return j,context,diag


class AnchorFlowEdge(Baseline):
    def __init__(self,pretrained=False,flow_steps=3,encoder='mobilenetv4_conv_small_050.e3000_r224_in1k',
                 model_name='v10_adaptive_jet',limited=True,**kwargs):
        if model_name!='v10_adaptive_jet' or flow_steps!=3: raise ValueError('V10 model contract')
        super().__init__(pretrained,3,encoder,'v9_metric_refine',limited)
        self.dynamics=BudgetedJetDynamics(limited,**kwargs)
