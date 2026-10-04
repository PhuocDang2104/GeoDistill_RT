"""Fast CPU contracts, no downloads/data/GPU. Numerical and training tests."""
import copy
import inspect
import json
import math
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from geometry import encode_jet,decode_jet,JetNODE
from geometry_primitives import translate,edge_weights,minmod
from ode_solver import solve,error_norm
from model import AnchorFlowEdge
from losses import objective,smooth_rmse
from loss_helpers import teacher_weights
import run

torch.set_num_threads(2)
CONFIG=json.loads(Path(__file__).with_name('config.json').read_text())

def batch(h=64,w=128):
    torch.manual_seed(7)
    rgb=torch.rand(1,3,h,w);sparse=torch.zeros(1,1,h,w);sparse[...,::4,::4]=10
    mask=(sparse>0).float();gt=torch.ones_like(sparse)*12
    valid=torch.zeros_like(sparse);valid[...,::5,::5]=1
    K=torch.tensor([[[700.,0.,w/2],[0.,700.,h/2],[0.,0.,1.]]])
    return {'rgb':rgb,'sparse':sparse,'mask':mask,'K':K,'gt':gt,'gt_mask':valid,
            'teacher':gt+1,'confidence':torch.ones_like(gt),
            'relative':20+rgb[:,:1]*10,'relative_confidence':torch.ones_like(gt)}

class SolverContracts(unittest.TestCase):
    def test_constant_field_reaches_fixed_horizon(self):
        y0=torch.ones(2,6,4,8)
        ys,r=solve(lambda t,y:torch.ones_like(y)*.3,y0)
        torch.testing.assert_close(ys[0],y0+.15,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(ys[1],y0+.3,atol=1e-6,rtol=1e-6)
        self.assertEqual(r['terminal_time'],1.);self.assertEqual(r['nfe'],13)
    def test_true_reject_and_tolerance_refinement(self):
        initial=torch.ones(1,6,2,4)
        errors=[];counts=[]
        for rtol,atol in ((.01,.001),(.0001,.00001)):
            ys,r=solve(lambda t,y:2*y,initial,rtol=rtol,atol=atol,max_nfe=769)
            errors.append(float((ys[-1]-math.exp(2)).abs().max()));counts.append(r['nfe'])
        self.assertLess(errors[1],errors[0]);self.assertGreater(counts[1],counts[0])
        self.assertGreater(r['rejected_steps'],0)
        self.assertEqual(r['nfe'],1+3*(r['accepted_steps']+r['rejected_steps']))
    def test_numerical_budget_is_not_forced_acceptance(self):
        with self.assertRaisesRegex(RuntimeError,'No forced acceptance'):
            solve(lambda t,y:torch.ones_like(y),torch.zeros(1,6,2,4),max_nfe=3)
    def test_time_dependent_field(self):
        y,r=solve(lambda t,y:t*torch.ones_like(y),torch.zeros(1,6,2,4))
        torch.testing.assert_close(y[-1],torch.ones_like(y[-1])*.5,atol=1e-6,rtol=1e-6)
        self.assertEqual(r['terminal_time'],1.)
    def test_gradient_to_initial_and_field(self):
        y0=torch.ones(1,6,2,4,requires_grad=True);a=torch.tensor(.3,requires_grad=True)
        ys,_=solve(lambda t,y:a*y,y0,rtol=.0001,atol=.00001)
        ys[-1].mean().backward()
        self.assertAlmostEqual(float(a.grad),math.exp(.3),places=3)
        self.assertTrue(torch.isfinite(y0.grad).all())
    def test_midpoint_exact_nfe(self):
        y,r=solve(lambda t,y:y,torch.ones(1,6,2,4),method='midpoint')
        self.assertEqual(r['nfe'],8);self.assertEqual(r['terminal_time'],1.)
        self.assertAlmostEqual(float(y[-1,0,0,0,0]),(1+.25+.5*.25**2)**4,places=5)
    def test_midpoint_matches_official_solver(self):
        import torchdiffeq
        y0=torch.ones(2,6,2,4);f=lambda t,y:.3*y+.2*t
        ours,_=solve(f,y0,method='midpoint')
        official=torchdiffeq.odeint(f,y0,torch.tensor([0.,.5,1.]),method='midpoint',options={'step_size':.25})[1:]
        torch.testing.assert_close(ours,official,rtol=1e-6,atol=1e-6)
    def test_worst_sample_channel_norm(self):
        x=torch.zeros(4,6,2,4);x[0,0]=2
        self.assertEqual(float(error_norm(x)),2.)
    def test_fp16_state_rejected(self):
        with self.assertRaisesRegex(ValueError,'FP32'):solve(lambda t,y:y,torch.zeros(1,6,2,4).half())

class GeometryContracts(unittest.TestCase):
    def test_chart_bounds_without_projection(self):
        z=torch.randn(2,6,8,16)*100;j=decode_jet(z)
        self.assertTrue(torch.isfinite(j).all())
        self.assertTrue((j[:,:1]>=1/120-1e-7).all() and (j[:,:1]<=10+1e-6).all())
        self.assertTrue((j[:,1:3].abs()<=.5*j[:,:1]+1e-7).all())
        self.assertTrue((j[:,3:].abs()<=.25*j[:,:1]+1e-7).all())
    def test_chart_interior_roundtrip(self):
        z=torch.randn(2,6,8,16)
        # Inverse tanh becomes ill-conditioned near saturated FP32 tails.
        torch.testing.assert_close(encode_jet(decode_jet(z)),z,atol=5e-5,rtol=1e-5)
    def test_chart_gradient_finite(self):
        z=torch.zeros(1,6,2,4,requires_grad=True);decode_jet(z).square().sum().backward()
        self.assertTrue(torch.isfinite(z.grad).all());self.assertGreater(float(z.grad.abs().sum()),0)
    def test_quadratic_transport_composition(self):
        j=torch.randn(1,6,8,16)
        torch.testing.assert_close(translate(translate(j,.2,.3),-.4,.5),translate(j,-.2,.8),atol=1e-6,rtol=1e-6)
    def test_zero_flux(self):
        w=edge_weights(torch.ones(1,2,8,16))
        self.assertEqual(float(w[:,0,...,-1].sum()),0);self.assertEqual(float(w[:,1,...,0].sum()),0)
        self.assertEqual(float(w[:,2,...,-1,:].sum()),0);self.assertEqual(float(w[:,3,...,0,:].sum()),0)
    def test_minmod(self):
        torch.testing.assert_close(minmod(torch.tensor([1.,-1.]),torch.tensor([-1.,-1.])),torch.tensor([0.,-1.]))

class ModelContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=run.make_model(CONFIG,torch.device('cpu'),False).eval();cls.b=batch()
    def output(self):
        with torch.no_grad():return self.model(*(self.b[k] for k in ('rgb','sparse','mask','K')))
    def test_parameter_count(self):self.assertEqual(sum(p.numel() for p in self.model.parameters()),582912)
    def test_no_learned_solver(self):
        self.assertFalse(any('step_head' in n or 'controller' in n or 'stop_head' in n for n,_ in self.model.named_parameters()))
    def test_teacher_not_input(self):
        self.assertEqual(list(inspect.signature(self.model.forward).parameters),['rgb','sparse','mask','K'])
    def test_feedback_and_actual_time_at_every_rhs_call(self):
        states=[];hook=self.model.dynamics.state.register_forward_pre_hook(lambda m,a:states.append(a[0].detach().clone()))
        self.output();hook.remove()
        self.assertEqual(len(states),self.model.dynamics.last_solver_report['nfe'])
        self.assertGreaterEqual(len(states),13)
        self.assertEqual(float(states[0][:,-1].max()),0)
        self.assertGreater(float(states[-1][:,-1].max()),.9)
        self.assertFalse(torch.equal(states[0][:,:6],states[1][:,:6]))
        self.assertTrue(all(x.dtype==torch.float32 for x in states))
    def test_fixed_time_observation_shapes(self):
        out=self.output()
        for name,s in (('D16',16),('D8',8),('D4',4),('D2',2),('D1',1),('D_full',1)):
            self.assertEqual(tuple(out[name].shape),(1,1,64//s,128//s))
        self.assertNotIn('D4_step3',out);self.assertEqual(float(out['dynamics_terminal_time_mean']),1.)
    def test_empty_invalid_sparse(self):
        b=self.b;s=b['sparse'].clone();s[...,0,0]=float('nan')
        with torch.no_grad():
            a=self.model(b['rgb'],s,b['mask'],b['K'])['D_full']
            e=self.model(b['rgb'],s*0,b['mask']*0,b['K'])['D_full']
        self.assertTrue(torch.isfinite(a).all() and torch.isfinite(e).all())
    def test_hard_diagnostic(self):
        out=self.output();m=self.b['mask'].bool();self.assertTrue(torch.equal(out['D_hard'][m],self.b['sparse'][m]))
    def test_diagnostics_same_depth(self):
        a=self.output()['D_full'];self.model.set_diagnostics(False);b=self.output()['D_full'];self.model.set_diagnostics(True)
        self.assertTrue(torch.equal(a,b))
    def test_no_full_resolution_cnn(self):
        sizes=[];handles=[]
        for mod in self.model.modules():
            if isinstance(mod,nn.Conv2d):handles.append(mod.register_forward_hook(lambda m,a,o:sizes.append(tuple(o.shape[-2:]))))
        self.output()
        for h in handles:h.remove()
        self.assertTrue(all(h<64 and w<128 for h,w in sizes))
    def test_fixed_mode_finite(self):
        old=self.model.dynamics.method
        try:
            self.model.dynamics.method='midpoint';out=self.output()
            self.assertTrue(torch.isfinite(out['D_full']).all());self.assertEqual(self.model.dynamics.last_solver_report['nfe'],8)
        finally:self.model.dynamics.method=old

class TrainingContracts(unittest.TestCase):
    def test_objective_backward_dual_teacher(self):
        m=run.make_model(CONFIG,torch.device('cpu'),False).train();m.freeze_encoder_bn();b=batch()
        mask,holdout=run.input_with_holdout(b,.1)
        out=m(b['rgb'],b['sparse'],mask,b['K']);loss,stats=objective(out,b,holdout,5,CONFIG)
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
        self.assertGreater(float(m.dynamics.reaction.weight.grad.abs().sum()),0)
        self.assertGreater(float(stats['kd_coverage']),0);self.assertGreater(float(stats['relative_pair_coverage']),0)
    def test_disabled_kd_and_ramps_zero(self):
        m=run.make_model(CONFIG,torch.device('cpu'),False).eval();b=batch()
        with torch.no_grad():p=m(*(b[k] for k in ('rgb','sparse','mask','K')))
        cfg={**CONFIG,'teacher_enabled':False,'relative_enabled':False}
        _,stats=objective(p,b,torch.zeros_like(b['mask']),0,cfg)
        for name in ('weighted_metric_kd','weighted_relative','weighted_log','weighted_edge','weighted_inverse_rmse','weighted_tail'):
            self.assertEqual(float(stats[name]),0.,name)
    def test_gt_and_sensor_exclusion_from_kd(self):
        b=batch();_,_,forbidden,eligible=teacher_weights(b,.5)
        self.assertEqual(float((eligible.float()*((b['gt_mask']>0)|(b['mask']>0)).float()).sum()),0)
    def test_inverse_definition(self):
        e=torch.tensor([[[[100.,-50.]]]])
        self.assertAlmostEqual(float(smooth_rmse(e,torch.ones_like(e))),math.sqrt(6250)-.001,places=4)
    def test_fresh_only(self):
        with self.assertRaisesRegex(ValueError,'fresh-only'):run.initialize_from_parent(None,{'init_checkpoint':'old.pth'})
    def test_early_stop(self):
        s={'monitor_best_rmse':None,'bad_epochs':0,'stopped':False}
        for e in range(20):s=run.early_stop_update(s,1.,e,CONFIG)
        self.assertTrue(s['stopped'])
    def test_t4_not_native_bf16(self):
        with patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.get_device_capability',return_value=(7,5)):
            self.assertFalse(run.native_bf16_supported());self.assertEqual(run.select_training_precision('auto'),'fp32')

if __name__=='__main__':unittest.main()
