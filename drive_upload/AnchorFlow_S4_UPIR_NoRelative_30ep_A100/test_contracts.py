"""Portable same-environment regression tests; no fixed cross-OS weight digest."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import experiment as E
from deployment import typed_deployment,export_onnx
from objective import loss_fn
torch.set_num_threads(2)
CFG=json.loads((E.ROOT/'config.json').read_text())

class Contracts(unittest.TestCase):
    def inputs(self):
        rgb=torch.rand(1,3,32,64);s=torch.full((1,1,32,64),20.)
        m=(torch.rand_like(s)>.9).float()
        K=torch.tensor([[[50.,0.,31.],[0.,50.,15.],[0.,0.,1.]]])
        return rgb,s,m,K

    def test_real_four_calls_no_fine_backward(self):
        n=E.make_model(CFG,E.VARIANTS[0]).eval();calls=[]
        h=n.dynamics.reaction.register_forward_hook(lambda *args:calls.append(1))
        o=n(*self.inputs());h.remove()
        self.assertEqual(len(calls),4);self.assertIsNone(n.fine_node)
        self.assertEqual(sum(p.numel() for p in n.parameters()),583880)
        self.assertEqual(n.detail1.pir.body[0].in_channels,41)
        self.assertFalse(any(k.endswith('jp_beta') for k in n.state_dict()))
        torch.testing.assert_close(o['D2'],o['D2_pre_fine'],atol=0,rtol=0)
        o['D_full'].mean().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in n.parameters()))

    def test_same_environment_shared_init_and_repeat(self):
        n=E.make_model(CFG,E.VARIANTS[0]).eval()
        E.seed_all(CFG['seed'])
        a=E.AnchorFlowEdge(pretrained=False,flow_steps=2,encoder=CFG['encoder'],
            model_name='v11_node',phase_context_enabled=True,fine_node_enabled=True,pir_enabled=True)
        a.dynamics.configure(CFG)
        self.assertEqual(E.common_weight_hash(n),E.common_weight_hash(a))
        self.assertEqual(E.common_weight_hash(n),E.common_weight_hash(E.make_model(CFG,E.VARIANTS[0])))
        self.assertEqual(float(n.detail1.pir.body[0].weight[:,40:].detach().abs().max()),0.)

    def test_zero_added_weights_match_old_pir(self):
        from phase_readout import PhaseInnovationResidual
        p=PhaseInnovationResidual()
        with torch.no_grad():p.head.weight.fill_(.02);p.head.bias[:4].fill_(.3)
        old=copy.deepcopy(p)
        p.enable_uncertainty_feature(.002)
        self.assertEqual(sum(x.numel() for x in p.parameters()),968)
        h=torch.rand(1,24,16,32);d=torch.full((1,1,32,64),20.);s=torch.zeros_like(d);m=torch.zeros_like(d)
        u=torch.rand(1,1,16,32)*.01
        expected=old(h,d,s,m);actual=p(h,d,s,m,u)
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
        actual[0].mean().backward()
        self.assertGreater(float(p.body[0].weight.grad[:,40:].abs().sum()),0.)

    def test_cue_gradient_and_zero_u_control(self):
        from phase_readout import PhaseInnovationResidual
        p=PhaseInnovationResidual();p.enable_uncertainty_feature()
        with torch.no_grad():p.body[0].weight[:,40:].fill_(.1);p.head.weight.fill_(.02);p.head.bias[:4].fill_(.3)
        h=torch.rand(1,24,16,32);d=torch.full((1,1,32,64),20.);s=torch.zeros_like(d);m=torch.zeros_like(d)
        u=torch.full((1,1,16,32),.002,requires_grad=True)
        a=p(h,d,s,m,u)[0];b=p(h,d,s,m,torch.zeros_like(u))[0]
        self.assertFalse(torch.equal(a,b));a.mean().backward()
        self.assertTrue(torch.isfinite(u.grad).all());self.assertGreater(float(u.grad.abs().sum()),0.)
        with self.assertRaises(ValueError):p(h,d,s,m,None)

    def test_inference_cue_survives_telemetry_off(self):
        n=E.make_model(CFG,E.VARIANTS[0]).eval();n.set_diagnostics(False)
        with torch.no_grad():n.detail1.pir.body[0].weight[:,40:].fill_(.1);n.detail1.pir.head.weight.fill_(.02)
        def cue(value):
            def hook(module,args,output):output[-1]['query_uncertainty']=torch.full_like(output[-1]['query_uncertainty'],value)
            return hook
        args=self.inputs();hook=n.phase2.register_forward_hook(cue(0.))
        a=n(*args);hook.remove();hook=n.phase2.register_forward_hook(cue(.002))
        b=n(*args);self.assertFalse(torch.equal(a['D_full'],b['D_full']))
        n.set_diagnostics(True);c=n(*args);hook.remove()
        torch.testing.assert_close(b['D_full'],c['D_full'],atol=0,rtol=0)

    def test_s4_times_and_gradient(self):
        from ode_solver import solve
        times=[]
        def f(t,z):times.append(float(t));return z
        z=torch.ones(1,6,2,2,requires_grad=True)
        traj,report=solve(f,z,method='midpoint',fixed_steps=2,max_nfe=4)
        self.assertEqual(times,[0.,.25,.5,.75]);self.assertEqual(report['nfe'],4)
        traj[-1].sum().backward();self.assertTrue(torch.isfinite(z.grad).all())

    def test_loss_backward_and_relative_logging(self):
        from relative_loss import relative_structure
        n=E.make_model(CFG,E.VARIANTS[0]).eval();rgb,s,m,K=self.inputs()
        m.zero_();o=n(rgb,s,m,K)
        batch=dict(rgb=rgb,sparse=s,mask=m,K=K,gt=torch.full_like(s,20.),gt_mask=torch.zeros_like(m),
            relative=torch.rand_like(s),relative_confidence=torch.ones_like(s))
        loss,stats=loss_fn(o,batch,torch.zeros_like(m),3.,dict(CFG,teacher_enabled=False,relative_enabled=True,relative_weight=.015))
        expected,_,coverage=relative_structure(o['D2'],batch['relative'],batch['relative_confidence'],rgb,m)
        torch.testing.assert_close(stats['relative_gradient'],expected.detach())
        torch.testing.assert_close(stats['relative_pair_coverage'],coverage.detach())
        torch.testing.assert_close(stats['weighted_relative'],expected.detach()*.015)
        self.assertGreater(float(stats['relative_gradient']),0.)
        self.assertTrue(torch.isfinite(loss));loss.backward()

    def test_fp32_onnx_after_backward_preserves_weights(self):
        import onnx
        n=E.make_model(CFG,E.VARIANTS[0]).eval()
        with torch.no_grad():n.detail1.pir.body[0].weight[:,40:].fill_(.1)
        args=self.inputs();n(*args)['D_full'].mean().backward()
        weights={k:v.detach().clone() for k,v in n.state_dict().items()}
        deploy=typed_deployment(n)
        for k,v in n.state_dict().items():torch.testing.assert_close(v,weights[k],atol=0,rtol=0)
        self.assertTrue(all(p.dtype==torch.float32 for p in deploy.parameters()))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'F.onnx';export_onnx(deploy,dict(zip(E.INPUTS,args)),path,22)
            g=onnx.load(str(path));onnx.checker.check_model(g,full_check=True)
            self.assertEqual(len(g.graph.input),4)
            self.assertFalse(any(t.data_type==onnx.TensorProto.BFLOAT16 for t in g.graph.initializer))
            self.assertTrue(any(list(t.dims)==[16,41,1,1] for t in g.graph.initializer))

    def test_recipe_gates_and_teacher_block(self):
        for k,v in [('epochs',26),('early_stopping',True),('ode_max_nfe',6),('upir_uncertainty_scale',.02),('tf32',True)]:
            with self.assertRaises(ValueError):E.assert_config(dict(CFG,**{k:v}))
        from data import KITTIDataset
        with self.assertRaises(ValueError):KITTIDataset(CFG,'val',teacher=True)
        for key,value in [('relative_enabled',True),('relative_weight',.015)]:
            with self.assertRaises(ValueError):E.assert_config(dict(CFG,**{key:value}))

    def test_default_never_calls_relative_or_requires_relative_tensors(self):
        import relative_loss
        n=E.make_model(CFG,E.VARIANTS[0]).eval();rgb,s,m,K=self.inputs()
        o=n(rgb,s,m,K)
        batch=dict(rgb=rgb,sparse=s,mask=m,K=K,gt=torch.full_like(s,20.),gt_mask=torch.ones_like(m),
            teacher=torch.full_like(s,20.),confidence=torch.ones_like(s))
        with patch.object(relative_loss,'relative_structure',side_effect=AssertionError('Disabled relative executed')) as relative:
            loss,stats=loss_fn(o,batch,torch.zeros_like(m),3.,CFG);relative.assert_not_called()
        for name in ('relative_gradient','relative_pair_coverage','relative_ramp','weighted_relative'):
            self.assertEqual(float(stats[name]),0.)
        self.assertTrue(torch.isfinite(loss));loss.backward()

    def test_prepare_does_not_read_relative_archive(self):
        import data,relative_data
        with tempfile.TemporaryDirectory() as folder:
            cfg=dict(CFG,work=folder,expected_subset_sha256=None)
            def metric_only(config):
                data.write_json(Path(folder)/'data_contract.json',{'teacher_report':{
                    'selected_teacher_content_sha256':'06bd17a84faa5020692873ee598498486d2030f588b7429321fa0a70d9ebb7a6'}})
            with patch.object(data,'prepare_metric',side_effect=metric_only) as metric,patch.object(relative_data,'extract_relative',side_effect=AssertionError('Disabled relative archive read')) as relative:
                data.prepare(cfg);metric.assert_called_once();relative.assert_not_called()
            self.assertNotIn('relative_teacher',json.loads((Path(folder)/'data_contract.json').read_text()))

    def test_comparison_only_relaxes_relative_metadata(self):
        from compare_history import metric_data_contract
        a={'train':1600,'val':400,'teacher_report':{'hash':'metric'},'relative_teacher':{'hash':'relative'},'relative_val_used':False,'min_non_gt_relative_pixels':1024}
        b={'train':1600,'val':400,'teacher_report':{'hash':'metric'}}
        self.assertEqual(metric_data_contract(a),metric_data_contract(b))
        b['val']=401
        self.assertNotEqual(metric_data_contract(a),metric_data_contract(b))

    def test_acceptance_does_not_fake_a_win(self):
        from compare_history import acceptance
        self.assertTrue(acceptance(dict(rmse_m=.99,irmse_km_inv=3.18,median_ms=3.80),CFG)['all_targets_met'])
        self.assertFalse(acceptance(dict(rmse_m=1.,irmse_km_inv=3.25,median_ms=3.85),CFG)['all_targets_met'])

    def test_historical_snapshot_comparison(self):
        import compare_history as C
        import run,sys
        with tempfile.TemporaryDirectory() as folder:
            cfg=dict(CFG,drive_runs=folder,run_name='new',historical_run=folder+'/absent',
                historical_d_run=folder+'/absent_d',historical_e_run=folder+'/absent_e')
            root=Path(folder)/'new';own=root/E.VARIANTS[0];own.mkdir(parents=True)
            ckpt={'contract':{'data':{'fixture':True}},'state':{'initial_shared_sha256':'init'}}
            report={'qualification_passed':True,'variant':E.VARIANTS[0],'checkpoint_epoch':29,
                'checkpoint_sha256':'test','trt_graph':{'all':{'rmse_m':.99,'irmse_km_inv':3.18,'mae_m':.2}},
                'trt_graph_boundary':{'bands':{'3':{'rmse_m':1.7}}},
                'timing':{'trt_graph':{'synchronized_wall':{'median_ms':3.80,'p95_ms':4.}}},
                'parameters':583880,'parity_gates':cfg['benchmark'],'training_epochs':30}
            (own/'trt_benchmark.json').write_text(json.dumps(report))
            with patch.object(torch,'load',return_value=ckpt),patch.object(run,'contract',return_value=ckpt['contract']),patch.object(C,'digest',return_value='test'),patch.dict(sys.modules,{'cv2':None}),patch('builtins.print'):
                C.compare_history(cfg)
            output=json.loads((root/'comparison.json').read_text())
            self.assertEqual(len(output['rows']),7);self.assertEqual(len(output['warnings']),5)
            self.assertFalse(output['rows'][0]['matched_training_verified'])
            self.assertEqual(output['rows'][5]['variant'],'G_s4_upir_norel')
            self.assertEqual(output['rows'][5]['relative_kd'],'off')
            self.assertTrue(output['acceptance']['all_targets_met'])

    def test_syntax_and_notebook(self):
        for p in E.ROOT.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
        for p in E.ROOT.glob('*.ipynb'):
            for cell in json.loads(p.read_text(encoding='utf-8'))['cells']:
                if cell['cell_type']=='code':ast.parse(''.join(cell['source']))

if __name__=='__main__':unittest.main()
