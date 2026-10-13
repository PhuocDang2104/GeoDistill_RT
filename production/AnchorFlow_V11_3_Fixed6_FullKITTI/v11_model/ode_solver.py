"""Numerical integration only: no NN, GT, teacher, learned h or task-stop.

Fixed6 production uses classic explicit RK3, two half-length steps, exactly
six RHS calls. Historical Bosh3/midpoint are retained for explicit audits only.
"""
from __future__ import annotations
import torch
from torch import nn


def error_norm(x):
    """Worst sample/channel spatial RMS; one numerical grid for the batch.

    Prevent a difficult sample/channel being diluted by the rest of the batch.
    The error is already divided by atol+rtol*max(abs(y0),abs(y1)).
    """
    return x.square().flatten(2).mean(2).sqrt().amax()


class CountedField(nn.Module):
    def __init__(self, field, max_nfe):
        super().__init__()
        self.field = field
        self.max_nfe = max_nfe
        self.nfe = 0
        self.accepted, self.rejected = [], 0
        self.final_time = None

    def forward(self, t, y):
        if self.nfe >= self.max_nfe:
            raise RuntimeError(f'NODE numerical budget exceeded ({self.max_nfe} NFE). '
                               'No forced acceptance, tolerance change or Euler fallback. '
                               'Inspect the field/solver report; use a NEW run for changed tolerances.')
        self.nfe += 1
        return self.field(t, y)

    def callback_accept_step(self, t, y, dt):
        self.accepted.append(dt.detach())
        self.final_time = (t + dt).detach()

    def callback_reject_step(self, t, y, dt):
        self.rejected += 1


def solve(field, initial, method='bosh3', rtol=.01, atol=.001,
          first_step=.25, max_step=.25, max_nfe=193, max_num_steps=64,
          fixed_steps=4, force_observation_steps=False):
    """Return states at t=.5 and t=1, not states after two solver steps.

    Solver decisions do not receive GT/task benefit. Finite budgets raise, never
    force an inaccurate acceptance or pretend the numerical tolerance was met.
    Direct autograd differentiates the executed solve, not the discrete choices.
    """
    if initial.dtype != torch.float32:
        raise ValueError('NODE state/integration must be FP32')
    if method not in ('bosh3', 'midpoint', 'rk3'):
        raise ValueError(f'Unknown solver: {method}')
    if not (0 < rtol and 0 < atol and 0 < first_step <= max_step <= 1.):
        raise ValueError('Invalid numerical solver parameters')
    counted = CountedField(field, max_nfe)
    if method == 'bosh3':
        import torchdiffeq
        if torchdiffeq.__version__ != '0.2.5':
            raise RuntimeError('Reproducible V11 requires torchdiffeq==0.2.5')
        times = initial.new_tensor([0., .5, 1.])
        # Observation times are NOT mandatory solver endpoints. Dense output
        # samples t=.5 even when a numerical step crosses it. Only T=1 is an
        # endpoint by default; the optional old policy is kept for matched audit.
        trajectory = torchdiffeq.odeint(counted, initial, times, method='bosh3',
            rtol=rtol, atol=atol, options={
                'first_step':first_step, 'max_step':max_step,
                'step_t':times[1:] if force_observation_steps else times[-1:],
                'max_num_steps':max_num_steps,
                'norm':error_norm})[1:]
    elif method == 'rk3':
        if fixed_steps != 2:
            raise ValueError('Fixed6 RK3 requires exactly two steps, T=1 and half-time observation')
        if max_nfe < 6:
            raise ValueError('Fixed6 RK3 requires a budget of at least six NFE')
        # Classic Kutta RK3: c=(0,1/2,1), b=(1/6,2/3,1/6).
        # No data-dependent mesh, rejection, dense interpolation or host scalar
        # reductions. State and all three stages remain differentiable FP32.
        y, states = initial, []
        dt = .5
        for k in range(2):
            t = initial.new_tensor(k * dt)
            k1 = counted(t, y)
            k2 = counted(t + .5*dt, y + .5*dt*k1)
            k3 = counted(t + dt, y - dt*k1 + 2*dt*k2)
            y = y + (dt/6)*(k1 + 4*k2 + k3)
            states.append(y)
        trajectory = torch.stack(states)
    else:
        if fixed_steps < 2 or fixed_steps % 2:
            raise ValueError('Fixed midpoint needs an even number of steps to sample t=.5')
        y, states = initial, []
        dt = 1. / fixed_steps
        for k in range(fixed_steps):
            t = initial.new_tensor(k * dt)
            k1 = counted(t, y)
            k2 = counted(t + .5*dt, y + .5*dt*k1)
            y = y + dt*k2
            if k+1 in (fixed_steps//2, fixed_steps):states.append(y)
            counted.callback_accept_step(t, y, initial.new_tensor(dt))
        trajectory = torch.stack(states)
    if method in ('midpoint', 'rk3'):
        # No .item()/CPU reductions in the static export path.
        report={'method':method,'nfe':counted.nfe,'accepted_steps':fixed_steps,
                'rejected_steps':0,'accepted_h_mean':1./fixed_steps,
                'accepted_h_min':1./fixed_steps,'accepted_h_max':1./fixed_steps,
                'terminal_time':1.,'tolerance_controlled':False}
        if method == 'rk3':
            report.update(order=3, stages_per_step=3, observation_times=[.5,1.])
    else:
        hs = torch.stack(counted.accepted).cpu()
        report={'method':method,'nfe':counted.nfe,'accepted_steps':len(counted.accepted),
                'rejected_steps':counted.rejected,'accepted_h_mean':float(hs.mean()),
                'accepted_h_min':float(hs.min()),'accepted_h_max':float(hs.max()),
                'terminal_time':float(counted.final_time.cpu()),
                'tolerance_controlled':True,'rtol':rtol,'atol':atol,
                'first_step':first_step,'max_step':max_step,
                'force_observation_steps':bool(force_observation_steps)}
        if abs(report['terminal_time']-1.) > 1e-6:
            raise RuntimeError(f'NODE did not terminate at fixed T=1: {report}')
    return trajectory, report
