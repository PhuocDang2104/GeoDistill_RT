import inspect
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from model import AnchorFlowEdge,AdaptiveJetController,BudgetedJetDynamics
from model_baseline import AnchorFlowEdge as Baseline
from losses import objective,stop_targets
from losses_baseline import objective as baseline_loss
from adaptive_report import AdaptiveMetrics
from model_v8 import translate
from data import KITTIDataset
import run

def config(): return json.loads(Path(__file__).with_name('config.json').read_text())

def fixture():
    inputs=run.sample_inputs('cpu',64,128)
    rgb,sparse,mask,K=inputs
    relative=torch.linspace(-1,1,128)[None,None,None].expand(1,1,64,128).clone()
    batch=dict(rgb=rgb,sparse=sparse,mask=mask,K=K,gt=torch.full_like(mask,25),
        gt_mask=(torch.rand_like(mask)<.15).float(),teacher=torch.full_like(mask,22),
        confidence=torch.full_like(mask,.8),relative=relative,relative_confidence=torch.ones_like(mask))
    return inputs,batch

class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_config_fresh40(self):
        c=config(); self.assertEqual(c['epochs'],40); self.assertIsNone(c['init_checkpoint'])
        self.assertTrue(c['encoder_pretrained'] and c['teacher_enabled'] and c['relative_enabled'])
        self.assertEqual(c['early_stop_min_epochs'],20); self.assertEqual(c['early_stop_patience'],8)
        self.assertFalse(c['cudnn_benchmark']); self.assertEqual(c['compute_weight'],0)
        self.assertEqual(list(inspect.signature(AnchorFlowEdge.forward).parameters),['self','rgb','sparse','mask','K'])

    def test_controller_size(self):
        c=AdaptiveJetController(); self.assertEqual(sum(p.numel() for p in c.parameters()),482)
        out=c(torch.randn(2,10)); self.assertTrue(torch.equal(out,torch.tensor([[4.,-2.],[4.,-2.]])))

    def test_bounds_and_horizon(self):
        x,_=fixture(); m=AnchorFlowEdge().eval()
        with torch.no_grad(): o=m(*x)
        self.assertTrue(((o['adaptive_h']>=1/6)&(o['adaptive_h']<=1/3)).all())
        self.assertTrue((o['adaptive_terminal_time']<=4/3).all())
        self.assertEqual(o['adaptive_executed_nfe'].item(),4)
        for k in range(1,5): self.assertTrue(torch.isfinite(o[f'D4_step{k}']).all())

    def test_forced_exit2_really_two_calls(self):
        x,_=fixture(); m=AnchorFlowEdge().eval()
        with torch.no_grad(): m.dynamics.controller.net[-1].bias[1]=20
        m.dynamics.mode='adaptive_batch1'
        calls=[]; handle=m.dynamics.reaction.register_forward_hook(lambda *args:calls.append(1))
        with torch.no_grad(): o=m(*x)
        handle.remove(); self.assertEqual(len(calls),2); self.assertEqual(o['adaptive_selected_nfe'].item(),2)

    def test_conditional_matches_masked_at_all_exits(self):
        x,_=fixture(); m=AnchorFlowEdge().eval()
        for threshold in (.001,.999):
            m.dynamics.stop_threshold=threshold; m.dynamics.mode='masked_adaptive'
            with torch.no_grad(): a=m(*x)
            m.dynamics.mode='adaptive_batch1'
            with torch.no_grad(): b=m(*x)
            self.assertTrue(torch.equal(a['D_full'],b['D_full']))
            self.assertTrue(torch.equal(a['adaptive_selected_nfe'],b['adaptive_executed_nfe']))
        # A time-only controller to stop specifically AFTER call three.
        class TimeController(torch.nn.Module):
            def forward(self,s): return torch.stack((s[:,9]*0+4,100*(s[:,9]-.8)),1)
        m.dynamics.controller=TimeController(); m.dynamics.stop_threshold=.5
        m.dynamics.mode='masked_adaptive'
        with torch.no_grad(): a=m(*x)
        m.dynamics.mode='adaptive_batch1'
        with torch.no_grad(): b=m(*x)
        self.assertEqual(b['adaptive_executed_nfe'].item(),3); self.assertTrue(torch.equal(a['D_full'],b['D_full']))

    def test_batch1_branch_guard(self):
        x,_=fixture(); m=AnchorFlowEdge().train(); m.dynamics.mode='adaptive_batch1'
        with self.assertRaises(ValueError): m(*x)

    def test_static4_is_not_early_exit(self):
        x,_=fixture(); m=AnchorFlowEdge().eval(); m.dynamics.mode='static4'
        with torch.no_grad():
            m.dynamics.controller.net[-1].bias[1]=20; o=m(*x)
        self.assertEqual(o['adaptive_selected_nfe'].item(),4)

    def test_fixed3_identical_loaded_baseline(self):
        x,b=fixture(); baseline=Baseline().eval(); m=AnchorFlowEdge(policy='fixed3').eval()
        loaded=m.load_state_dict(baseline.state_dict(),strict=False)
        self.assertTrue(all(k.startswith('dynamics.controller.') for k in loaded.missing_keys))
        self.assertFalse(loaded.unexpected_keys)
        with torch.no_grad(): a=baseline(*x); o=m(*x)
        for key in ('D4','D2','D1','D_full'): self.assertTrue(torch.allclose(a[key],o[key],atol=1e-5,rtol=1e-6),key)
        c={**config(),'integration_policy':'fixed3'}
        loss,_=objective(o,b,b['mask']*0,4,c)
        expected,_=baseline_loss(a,b,b['mask']*0,4,{**c,'model_name':'v9_metric_refine'})
        self.assertTrue(torch.allclose(loss,expected,atol=1e-5))

    def test_fixed4_and_learned4(self):
        x,_=fixture()
        for policy in ('fixed4','learned4'):
            m=AnchorFlowEdge(policy=policy).eval()
            with torch.no_grad(): o=m(*x)
            self.assertEqual(o['adaptive_selected_nfe'].item(),4)
            if policy=='fixed4': self.assertTrue(torch.allclose(o['adaptive_h'],torch.full((1,4),1/3)))

    def test_time_is_actual_accumulation(self):
        x,_=fixture(); m=AnchorFlowEdge().eval(); times=[]
        original=m.dynamics.vector_field
        def observed(*args): times.append(args[-1].clone()); return original(*args)
        m.dynamics.vector_field=observed
        with torch.no_grad(): o=m(*x)
        self.assertEqual(len(times),4); self.assertEqual(times[0].item(),0)
        for k in range(1,4): self.assertTrue(torch.equal(times[k],o['adaptive_h'][:,:k].sum(1)))

    def test_no_gt_or_teacher_in_controller(self):
        self.assertNotIn('gt',inspect.signature(BudgetedJetDynamics.statistics).parameters)
        self.assertNotIn('teacher',inspect.signature(BudgetedJetDynamics.forward).parameters)
        with self.assertRaises(ValueError): KITTIDataset(config(),'val',teacher=True)

    def test_finite_backward_fp32_bf16(self):
        for dtype in (None,torch.bfloat16):
            x,b=fixture(); m=AnchorFlowEdge().train(); m.freeze_encoder_bn()
            with torch.autocast('cpu',dtype=dtype or torch.bfloat16,enabled=dtype is not None): o=m(*x)
            loss,stats=objective(o,b,b['mask']*0,4,config()); loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
            self.assertGreater(m.dynamics.reaction.weight.grad.abs().sum().item(),0)
            self.assertGreater(m.dynamics.controller.net[-1].bias.grad.abs().sum().item(),0)
            self.assertGreater(m.phase2.metric[-1].weight.grad.abs().sum().item(),0)
            self.assertGreater(stats['relative_pair_coverage'].item(),0)
            self.assertEqual(stats['executed_nfe'].item(),4)

    def test_no_masking_empty_sparse(self):
        x,_=fixture(); rgb,s,m,K=x
        with torch.no_grad(): o=AnchorFlowEdge().eval()(rgb,s*0,m*0,K)
        self.assertTrue(torch.isfinite(o['D_full']).all())

    def test_labels_future_benefit_detached(self):
        b={'gt':torch.full((2,1,4,4),10.),'gt_mask':torch.ones(2,1,4,4)}
        p={'D4':torch.zeros(2,1,4,4)}
        for k,d in enumerate((14.,13.,12.,15.),1): p[f'D4_step{k}']=torch.full((2,1,4,4),d,requires_grad=True)
        labels,benefit,valid=stop_targets(p,b)
        self.assertTrue(torch.equal(labels,torch.tensor([[0.,1.],[0.,1.]])))
        self.assertFalse(benefit.requires_grad)
        b['gt_mask']*=0; _,_,valid=stop_targets(p,b); self.assertEqual(valid.sum().item(),0)

    def test_disabled_relative_and_ramps_are_exact(self):
        x,b=fixture(); m=AnchorFlowEdge().eval()
        with torch.no_grad(): o=m(*x)
        c={**config(),'relative_enabled':False,'teacher_enabled':False}
        loss,stats=objective(o,b,b['mask']*0,4,c)
        self.assertEqual(stats['weighted_relative'].item(),0)
        self.assertEqual(stats['weighted_metric_kd'].item(),0)
        self.assertEqual(stats['relative_ramp'].item(),1)
        self.assertEqual(stats['tail_ramp'].item(),1)
        self.assertTrue(torch.isfinite(loss))

    def test_geometric_transport_exact(self):
        j=torch.tensor([.1,.01,.02,.003,.004,.005])[None,:,None,None]
        recovered=translate(translate(j,1.,-2.),-1.,2.)
        self.assertTrue(torch.allclose(recovered,j,atol=1e-7))

    def test_projection_unchanged(self):
        j=torch.randn(2,6,8,8)
        p=BudgetedJetDynamics.project(j)
        self.assertTrue(((p[:,:1]>=1/120)&(p[:,:1]<=10)).all())
        self.assertTrue((p[:,1:3].abs()<=.5*p[:,:1]).all())
        self.assertTrue((p[:,3:].abs()<=.25*p[:,:1]).all())

    def test_native_bf16_guard_on_t4(self):
        with patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.get_device_capability',return_value=(7,5)),patch('torch.cuda.is_bf16_supported',return_value=True):
            self.assertFalse(run.native_bf16_supported()); self.assertEqual(run.select_training_precision('auto'),'fp32')
            with self.assertRaises(RuntimeError): run.device_setup(config())

    def test_auto_bf16_native(self):
        with patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.get_device_capability',return_value=(8,0)),patch('torch.cuda.is_bf16_supported',return_value=True) as supported:
            self.assertEqual(run.select_training_precision('auto'),'bf16')
            supported.assert_called_with(including_emulation=False)

    def test_early_stop_not_before20(self):
        c=config(); s={'monitor_best_rmse':1.,'bad_epochs':0,'stopped':False}
        for k in range(20):
            s=run.early_stop_update(s,1.1,k,c)
            if k<19: self.assertFalse(s['stopped'])
        self.assertTrue(s['stopped'])

    def test_diagnostics_summary(self):
        x,b=fixture(); m=AnchorFlowEdge().eval()
        with torch.no_grad(): o=m(*x)
        a=AdaptiveMetrics(torch.device('cpu'),config()); a.update(o,b); r=a.report()
        self.assertEqual(r['samples'],1); self.assertEqual(r['executed_nfe_mean'],4)
        self.assertIn('D4_step4',r['quarter_trajectory'])
        self.assertEqual(sum(r['exit_fraction'].values()),1)

    def test_training_loop_checkpoint_resume_and_protocol_guard(self):
        # Synthetic small images / local temporary run, not a GPU accuracy run.
        _,b=fixture()
        row={k:(v[0] if torch.is_tensor(v) else v) for k,v in b.items()}
        def loader(c,split,generator=None,batch_size=None):
            sample=row if split=='train' else {k:v for k,v in row.items() if k not in ('teacher','confidence','relative','relative_confidence')}
            return torch.utils.data.DataLoader([sample]*2,batch_size=1,shuffle=split=='train',generator=generator)
        with tempfile.TemporaryDirectory(prefix='v10_train_contract_') as directory:
            base=Path(directory);work=base/'work';work.mkdir()
            (work/'data_contract.json').write_text(json.dumps({'fixture':True,'train':2,'val':2,'test':0}))
            c={**config(),'work':str(work),'drive_runs':str(base/'drive'),'run_name':'fixture',
               'amp':'fp32','encoder_pretrained':False,'epochs':2,'workers':0,'batch_size':1,'fused_adamw':False}
            with patch.object(run,'device_setup',return_value=torch.device('cpu')),patch.object(run,'data_loader',side_effect=loader),patch('torch.cuda.get_device_name',return_value='CPU contract fixture'),contextlib.redirect_stdout(io.StringIO()):
                run.train(c,'dual_teacher')
                dest=base/'drive/fixture/dual_teacher'
                status=json.loads((dest/'training_status.json').read_text())
                self.assertEqual(status['epochs_completed'],2)
                checkpoint=(dest/'last.pth').read_bytes()
                run.train(c,'dual_teacher')
                self.assertEqual((dest/'last.pth').read_bytes(),checkpoint)
                with self.assertRaisesRegex(RuntimeError,'Resume protocol changed'):
                    run.train({**c,'step_min':.2},'dual_teacher')
                self.assertEqual((dest/'last.pth').read_bytes(),checkpoint)

if __name__=='__main__': unittest.main()
