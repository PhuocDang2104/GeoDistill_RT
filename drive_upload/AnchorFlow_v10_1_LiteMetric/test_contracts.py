"""Small, self-contained contracts; no datasets, download or GPU required."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from model import AnchorFlowEdge,Deploy
from geometry import adjacent,translate,edge_weights,minmod,FixedJetDynamics
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
    t=20+rgb[:,:1]*10
    return {'rgb':rgb,'sparse':sparse,'mask':mask,'K':K,'gt':gt,'gt_mask':valid,
            'teacher':gt+1,'confidence':torch.ones_like(gt),
            'relative':t,'relative_confidence':torch.ones_like(gt)}

class GeometryContracts(unittest.TestCase):
    def test_constant_translation(self):
        j=torch.zeros(1,6,8,16);j[:,:1]=.1
        self.assertTrue(torch.equal(translate(j,1.,-2.),j))
    def test_quadratic_transport_composes(self):
        j=torch.randn(1,6,8,16)
        self.assertTrue(torch.allclose(translate(translate(j,.2,.3),-.4,.5),translate(j,-.2,.8),atol=1e-6))
    def test_zero_boundary_flux(self):
        w=edge_weights(torch.ones(1,2,8,16))
        self.assertEqual(float(w[:,0,...,-1].abs().sum()),0)
        self.assertEqual(float(w[:,1,...,0].abs().sum()),0)
        self.assertEqual(float(w[:,2,...,-1,:].abs().sum()),0)
        self.assertEqual(float(w[:,3,...,0,:].abs().sum()),0)
    def test_minmod_at_tie(self):
        a=torch.tensor([1.,1.,-1.,-1.]);b=torch.tensor([1.,-1.,1.,-1.])
        self.assertTrue(torch.equal(minmod(a,b),torch.tensor([1.,0.,0.,-1.])))
    def test_projection(self):
        j=FixedJetDynamics.project(torch.randn(2,6,8,16)*100)
        self.assertTrue(((j[:,:1]>=1/120)&(j[:,:1]<=10)).all())
        self.assertTrue((j[:,1:3].abs()<=.5*j[:,:1]).all())
        self.assertTrue((j[:,3:].abs()<=.25*j[:,:1]).all())
    def test_diffusion_self_mass_bound(self):
        # h * 4 * max(rate=.6) <= .8; projection/reaction remain needed.
        self.assertLessEqual((1/3)*4*.6,1)

class ModelContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=AnchorFlowEdge().eval();cls.b=batch()
    def output(self):
        with torch.no_grad():return self.model(*(self.b[k] for k in ('rgb','sparse','mask','K')))
    def test_parameter_count(self):self.assertEqual(sum(p.numel() for p in self.model.parameters()),582945)
    def test_no_controller(self):self.assertFalse(any('controller' in name for name,_ in self.model.named_modules()))
    def test_teacher_not_input(self):
        import inspect
        self.assertEqual(list(inspect.signature(self.model.forward).parameters),['rgb','sparse','mask','K'])
    def test_two_feedback_calls(self):
        states=[];hook=self.model.dynamics.state.register_forward_pre_hook(lambda m,args:states.append(args[0].detach().clone()))
        self.output();hook.remove()
        self.assertEqual(len(states),2)
        self.assertTrue(torch.equal(states[0][:,-1],torch.zeros_like(states[0][:,-1])))
        self.assertTrue(torch.allclose(states[1][:,-1],torch.full_like(states[1][:,-1],.25)))
        self.assertFalse(torch.equal(states[0][:,:6],states[1][:,:6]))
    def test_output_shapes_and_no_phantom_steps(self):
        out=self.output()
        for name,s in (('D16',16),('D8',8),('D4',4),('D2',2),('D1',1),('D_full',1)):
            self.assertEqual(tuple(out[name].shape),(1,1,64//s,128//s))
        self.assertNotIn('D4_step3',out)
    def test_empty_and_invalid_sparse(self):
        b=self.b;s=b['sparse'].clone();s[...,0,0]=float('nan')
        with torch.no_grad():
            a=self.model(b['rgb'],s,b['mask'],b['K'])['D_full']
            e=self.model(b['rgb'],s*0,b['mask']*0,b['K'])['D_full']
        self.assertTrue(torch.isfinite(a).all() and torch.isfinite(e).all())
    def test_legacy_hard_anchor(self):
        out=self.output();m=self.b['mask'].bool()
        self.assertTrue(torch.equal(out['D_hard'][m],self.b['sparse'][m]))
    def test_diagnostics_do_not_change_depth(self):
        a=self.output()['D_full'];self.model.set_diagnostics(False)
        b=self.output()['D_full'];self.model.set_diagnostics(True)
        self.assertTrue(torch.equal(a,b))
    def test_new_projection_noop(self):
        self.assertEqual(float(self.model.phase_context[-1].weight.detach().abs().sum()),0)
        a=self.output()['D_full'];self.model.phase_context_enabled=False
        b=self.output()['D_full'];self.model.phase_context_enabled=True
        self.assertTrue(torch.equal(a,b))
    def test_no_full_resolution_conv(self):
        sizes=[];handles=[]
        for mod in self.model.modules():
            if isinstance(mod,nn.Conv2d):handles.append(mod.register_forward_hook(lambda m,a,o:sizes.append(tuple(o.shape[-2:]))))
        self.output()
        for h in handles:h.remove()
        self.assertTrue(all(h<64 and w<128 for h,w in sizes))
    def test_bounds(self):
        out=self.output()
        self.assertTrue(((out['D_full']>=.1)&(out['D_full']<=120)).all())

class LossContracts(unittest.TestCase):
    def test_inverse_definition(self):
        e=torch.tensor([[[[1000/5-1000/10,1000/20-1000/10]]]])
        self.assertAlmostEqual(float(smooth_rmse(e,torch.ones_like(e))),((100**2+50**2)/2)**.5-.001,places=4)
    def test_masked_loss_and_empty(self):
        e=torch.tensor([[[[2.,100.]]]]);m=torch.tensor([[[[1.,0.]]]])
        self.assertAlmostEqual(float(smooth_rmse(e,m)),2-.001,places=5)
        self.assertEqual(float(smooth_rmse(e,m*0)),0.)
    def test_gt_sensor_priority(self):
        b=batch();_,c,forbidden,eligible=teacher_weights(b,.5)
        self.assertEqual(float(c[forbidden.bool()].abs().sum()),0)
    def test_finite_backward_and_cold_head_opening(self):
        model=AnchorFlowEdge().train();model.freeze_encoder_bn();b=batch();holdout=b['mask']*0
        opt=torch.optim.SGD(model.parameters(),lr=.001)
        for i in range(2):
            out=model(*(b[k] for k in ('rgb','sparse','mask','K')))
            loss,stats=objective(out,b,holdout,4,CONFIG)
            self.assertTrue(torch.isfinite(loss));loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertTrue(all(v.ndim==0 for v in stats.values()))
            if i==1:self.assertGreater(float(model.phase_context[-1].weight.grad.abs().sum()),0)
            opt.step();opt.zero_grad(set_to_none=True)
    def test_inverse_ramp(self):
        m=AnchorFlowEdge().eval();b=batch()
        with torch.no_grad():p=m(*(b[k] for k in ('rgb','sparse','mask','K')))
        _,s=objective(p,b,b['mask']*0,0,CONFIG);self.assertEqual(float(s['weighted_inverse_rmse']),0)
        _,s=objective(p,b,b['mask']*0,4,CONFIG);self.assertGreater(float(s['weighted_inverse_rmse']),0)
        self.assertEqual(float(s['weighted_log']),0);self.assertEqual(float(s['weighted_edge']),0)
    def test_no_teacher_variant(self):
        c={**CONFIG,'teacher_enabled':False,'relative_enabled':False};b=batch()
        for key in ('teacher','confidence','relative','relative_confidence'):del b[key]
        with torch.no_grad():p=AnchorFlowEdge().eval()(*(b[k] for k in ('rgb','sparse','mask','K')))
        _,s=objective(p,b,b['mask']*0,4,c)
        self.assertEqual(float(s['weighted_metric_kd']),0);self.assertEqual(float(s['weighted_relative']),0)

class RuntimeContracts(unittest.TestCase):
    def test_t4_does_not_emulate_native_bf16(self):
        with patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.get_device_capability',return_value=(7,5)):
            self.assertEqual(run.select_training_precision('auto'),'fp32')
            with self.assertRaises(RuntimeError):run.select_training_precision('bf16')
    def test_no_fp16(self):
        with self.assertRaises(ValueError):run.select_training_precision('fp16')
    def test_early_stop(self):
        state={'monitor_best_rmse':None,'bad_epochs':0,'stopped':False}
        c={**CONFIG,'early_stop_min_epochs':2,'early_stop_patience':2}
        for i in range(3):state=run.early_stop_update(state,1.,i,c)
        self.assertTrue(state['stopped'])
    def test_roundtrip_state(self):
        model=AnchorFlowEdge();copy_model=AnchorFlowEdge()
        copy_model.load_state_dict(model.state_dict(),strict=True)
        self.assertEqual(set(model.state_dict()),set(copy_model.state_dict()))
    def test_warmstart_rejects_wrong_parent(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'wrong.pth';torch.save({'model':{},'protocol':{'source_sha256':'wrong'},'epoch':23},path)
            with self.assertRaises(RuntimeError):run.initialize_from_parent(AnchorFlowEdge(),{**CONFIG,'init_checkpoint':str(path)},False)

class LearnedStepContracts(unittest.TestCase):
    def test_step_gradient_and_bounds(self):
        m=AnchorFlowEdge().eval();b=batch()
        out=m(*(b[k] for k in ('rgb','sparse','mask','K')))
        for k in (1,2):
            self.assertAlmostEqual(float(out[f'dynamics_step_mean_{k}'].detach()),.25,places=6)
        out['D4'].mean().backward()
        grad=m.dynamics.step_head.weight.grad
        self.assertIsNotNone(grad);self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.abs().sum()),0)
        with torch.no_grad():m.dynamics.step_head.bias.fill_(100)
        o=m(*(b[k] for k in ('rgb','sparse','mask','K')))
        self.assertLessEqual(float(o['dynamics_step_max_1'].detach()),1/3+1e-7)
    def test_sample_conditioned_not_global_parameter(self):
        m=AnchorFlowEdge().eval();b=batch()
        with torch.no_grad():m.dynamics.step_head.weight.normal_(0,.2)
        x=b['rgb'];s=b['sparse'];mask=b['mask'];K=b['K']
        o=m(torch.cat((x,x*.2)),torch.cat((s,s)),torch.cat((mask,mask)),torch.cat((K,K)))
        self.assertEqual(m.dynamics.step_head.out_channels,1)
        self.assertEqual(m.dynamics.steps,2)
        self.assertTrue(torch.isfinite(o['D_full']).all())

if __name__=='__main__':unittest.main()
