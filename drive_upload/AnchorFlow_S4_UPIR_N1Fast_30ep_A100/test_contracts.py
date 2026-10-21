import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
import experiment as E
import history as H
from deployment import typed_deployment,export_onnx
import fast_ops as O
from benchmark_fast import target_checks

torch.set_num_threads(2)
CFG=json.loads((E.ROOT/'config.json').read_text())


def args():
    torch.manual_seed(127)
    rgb=torch.rand(1,3,32,64);mask=(torch.rand(1,1,32,64)>.85).float()
    sparse=(10+50*torch.rand_like(mask))*mask
    return rgb,sparse,mask,torch.tensor([[[50.,0.,31.],[0.,50.,15.],[0.,0.,1.]]])


def active(model):
    # Synthetic evaluation probe only, never used in trained GPU benchmark.
    with torch.no_grad():
        model.phase2.metric[-1].weight.normal_(std=.001)
        model.detail1.pir.head.weight.normal_(std=.01)
        model.detail1.pir.body[0].weight[:,40:].fill_(.1)
        model.dynamics.context[1].body[2][0].weight[:,64:].normal_(std=.001)
        model.decoder.fuse[0].refine.body[2][0].weight[:,96:].normal_(std=.001)
    return model


class Contracts(unittest.TestCase):
    def test_architecture_and_original_training_path(self):
        model=E.make_model(CFG,E.VARIANTS[0]);profile=E.architecture(model)
        self.assertEqual(E.VARIANTS,('Q1_s4_n1_fast',));self.assertEqual(E.TRAIN_VARIANTS,E.VARIANTS)
        self.assertEqual(profile['parameters'],582888)
        self.assertEqual(model.decoder.fuse[0].refine.body[0][0].out_channels,512)
        self.assertEqual(model.dynamics.context[1].body[0][0].out_channels,128)
        self.assertIsNone(model.fine_node)
        self.assertNotIsInstance(model.up4_2,O.FastUpsample)
        self.assertNotIsInstance(model.phase2,O.FastReadout)
        self.assertEqual(model.detail1.pir.body[0].in_channels,41)

    def test_neighbours_match_onehot_conv_at_all_borders_and_gradients(self):
        for h,w in ((1,1),(2,3),(16,32)):
            x=torch.rand(2,1,h,w,requires_grad=True)
            kernel=torch.eye(9).reshape(9,1,3,3)
            reference=F.conv2d(F.pad(x,(1,1,1,1),mode='replicate'),kernel)
            actual=O.neighbours9(x)
            torch.testing.assert_close(actual,reference,atol=0,rtol=0)
            a=torch.autograd.grad(actual.sum(),x,retain_graph=True)[0]
            b=torch.autograd.grad(reference.sum(),x)[0]
            torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_grouped_pool_keeps_original_2d_accumulation(self):
        for h,w in ((2,3),(16,32)):
            x=torch.randn(2,3,h,w)
            torch.testing.assert_close(O.pooled7(x),F.avg_pool2d(x,7,1,3),atol=0,rtol=0)

    def test_cardinal_quadratic_translate_and_gradients(self):
        from geometry_primitives import translate
        j=torch.randn(2,4,6,8,16,requires_grad=True)
        dx=torch.tensor([-1.,1.,0.,0.]).view(1,4,1,1,1)
        dy=torch.tensor([0.,0.,-1.,1.]).view(1,4,1,1,1)
        expected=translate(j,dx,dy);actual=O.cardinal_translate(j)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        a=torch.autograd.grad(actual.square().sum(),j,retain_graph=True)[0]
        b=torch.autograd.grad(expected.square().sum(),j)[0]
        torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_moment_pooling_and_empty_support(self):
        from model_base import sparse_innovation
        from phase_readout import phase_innovation
        _,s,m,_=args();d=10+torch.rand_like(s)*10
        for mask in (m,torch.zeros_like(m)):
            with torch.no_grad():
                a,b=O.sparse_innovation(d,s,mask),sparse_innovation(d,s,mask)
                torch.testing.assert_close(a,b,atol=1e-5,rtol=2e-5)
                aa,ab=O.phase_innovation(d,s,mask);ba,bb=phase_innovation(d,s,mask)
                torch.testing.assert_close(aa,ba,atol=0,rtol=0)
                torch.testing.assert_close(ab,bb,atol=0,rtol=0)
                self.assertTrue(torch.isfinite(a).all() and torch.isfinite(aa).all())
                if not mask.any():self.assertEqual(float(a.abs().sum()),0)

    def test_copy_only_weights_buffers_and_original_classes_untouched(self):
        model=active(E.make_model(CFG,E.VARIANTS[0]).eval())
        state={k:v.clone() for k,v in model.state_dict().items()}
        fast=typed_deployment(model,fast=True);original=typed_deployment(model,fast=False)
        self.assertIsInstance(fast.model.up4_2,O.FastUpsample)
        self.assertNotIsInstance(model.up4_2,O.FastUpsample)
        self.assertNotIsInstance(original.model.phase2,O.FastReadout)
        self.assertEqual(sum(p.numel() for p in fast.parameters()),582888)
        for key,value in state.items():
            torch.testing.assert_close(model.state_dict()[key],value,atol=0,rtol=0)
            torch.testing.assert_close(fast.model.state_dict()[key],value,atol=0,rtol=0)

    def test_active_whole_model_original_vs_fast_input_freshness(self):
        model=active(E.make_model(CFG,E.VARIANTS[0]).eval())
        original=typed_deployment(model,fast=False);fast=typed_deployment(model,fast=True)
        a=args();b=(a[0]*.7+.1,torch.zeros_like(a[1]),torch.zeros_like(a[2]),a[3]*1.1)
        with torch.no_grad():
            first=None
            for batch in (a,b,a):
                ref=original(*batch);value=fast(*batch);error=(ref-value).abs()
                self.assertLessEqual(float(error.mean()),.001);self.assertLessEqual(float(error.max()),.06)
                if first is None:first=value.clone()
            torch.testing.assert_close(first,value,atol=0,rtol=0)

    def test_original_backward_four_actual_rhs_context_once(self):
        model=E.make_model(CFG,E.VARIANTS[0]).eval();calls=[];contexts=[]
        h=model.dynamics.reaction.register_forward_hook(lambda *x:calls.append(1))
        c=model.dynamics.context.register_forward_hook(lambda *x:contexts.append(1))
        out=model(*args());h.remove();c.remove();out['D_full'].mean().backward()
        self.assertEqual((len(calls),len(contexts)),(4,1))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        torch.testing.assert_close(out['D2'],out['D2_pre_fine'],atol=0,rtol=0)

    def test_parent_constructor_and_training_ast_are_unchanged(self):
        policy=json.loads((E.ROOT/'evidence/parent_policy.json').read_text())
        for file,key in (('experiment.py','constructor_ast'),('run.py','training_function_ast')):
            functions={n.name:ast.dump(n,include_attributes=False) for n in ast.parse((E.ROOT/file).read_text()).body if isinstance(n,ast.FunctionDef)}
            for name,expected in policy[key].items():
                self.assertEqual(H.canonical_ast_dump(functions[name]),H.canonical_ast_dump(expected),name)

    def test_same_fresh_full_weights_and_historical_hash(self):
        a=E.make_model(CFG,E.VARIANTS[0]);b=E.make_model(CFG,E.VARIANTS[0])
        self.assertEqual(H.full_weight_hash(a),H.full_weight_hash(b))
        self.assertEqual(E.common_weight_hash(a),E.common_weight_hash(b))

    def test_config_relative_fixed_gates_and_no_tolerance_relaxation(self):
        E.assert_config(CFG)
        self.assertFalse(CFG['relative_enabled']);self.assertEqual(CFG['relative_weight'],0)
        self.assertTrue(CFG['teacher_enabled']);self.assertEqual(CFG['epochs'],30)
        self.assertEqual((CFG['benchmark']['pixel_mean_atol_m'],CFG['benchmark']['pixel_max_atol_m']),(.001,.06))
        for key,value in (('relative_enabled',True),('epochs',26),('tf32',True)):
            bad=copy.deepcopy(CFG);bad[key]=value
            with self.assertRaises(ValueError):E.assert_config(bad)
        bad=copy.deepcopy(CFG);bad['deployment_optimization']['fast_builder_level']=4
        with self.assertRaises(ValueError):E.assert_config(bad)

    def test_recipe_scoped_metadata_not_teacher_or_loss(self):
        a=copy.deepcopy(CFG);a['run_name']='old';a.pop('deployment_optimization')
        self.assertEqual(H.recipe(a),H.recipe(CFG))
        a['kd_weight']*=2;self.assertNotEqual(H.recipe(a),H.recipe(CFG))
        a={'teacher_report':{'signature':{'archive_mtime_ns':1,'sha256':'x'}}}
        b=copy.deepcopy(a);b['teacher_report']['signature']['archive_mtime_ns']=2
        self.assertEqual(H.comparable_data(a),H.comparable_data(b))
        b['teacher_report']['signature']['sha256']='z';self.assertNotEqual(H.comparable_data(a),H.comparable_data(b))

    def test_targets_strict_unrounded_and_no_promise(self):
        score={'rmse_m':.979622,'irmse_km_inv':3.180601};wall={'mean_ms':3.8999}
        self.assertTrue(target_checks(score,wall,CFG)['all_passed'])
        self.assertFalse(target_checks(dict(score,rmse_m=.98),wall,CFG)['all_passed'])
        self.assertFalse(target_checks(dict(score,irmse_km_inv=3.2),wall,CFG)['all_passed'])
        self.assertFalse(target_checks(score,{'mean_ms':3.90001},CFG)['all_passed'])
        self.assertFalse(target_checks(dict(score,rmse_m=float('nan')),wall,CFG)['all_passed'])

    def test_completed30_parent_best_checkpoint_can_be_epoch27(self):
        cfg=copy.deepcopy(CFG);data={'split':'fixture'}
        contract={'variant':E.REFERENCE_VARIANT,'config':cfg,'data':data,'source_sha256':'old'}
        checkpoint={'contract':contract,'state':{'initial_shared_sha256':'init','rows':[{}]*28}}
        images=[{'sample_id':str(i),'pixels':63562+(i<192)} for i in range(400)]
        report={'variant':E.REFERENCE_VARIANT,'training_contract':contract,'qualification_passed':True,
                'training_epochs':30,'parameters':582888,'inference_precision':'explicit_FP32_TF32_OFF',
                'initial_shared_sha256':'init','per_image':images}
        self.assertTrue(H.validate_parent(report,checkpoint,{'completed_epochs':30},cfg,data,{'source_sha256':'old'}))
        with self.assertRaises(RuntimeError):H.validate_parent(report,checkpoint,{'completed_epochs':28},cfg,data,{'source_sha256':'old'})

    def test_full_comparison_pipeline_and_stale_session_rejection(self):
        from benchmark_fast import compare
        with tempfile.TemporaryDirectory() as tmp:
            cfg=copy.deepcopy(CFG);cfg['drive_runs']=tmp;cfg['run_name']='run'
            root=Path(tmp)/'run';(root/'preflight_N1').mkdir(parents=True);(root/E.VARIANTS[0]).mkdir()
            (root/'benchmark_session.json').write_text(json.dumps({'session_id':'session'}))
            wall={'mean_ms':3.88,'median_ms':3.87,'p95_ms':3.91}
            timing={'warmup':100,'iterations':300,'distinct_images':100,'synchronized_wall':wall}
            score={'rmse_m':.9796,'irmse_km_inv':3.18,'mae_m':.257,'pixels':25424992}
            images=[{'sample_id':str(i),'pixels':63562+(i<192)} for i in range(400)]
            report={'variant':E.REFERENCE_VARIANT,'qualification_passed':True,'benchmark_session_id':'session',
                'source_sha256':E.source_hash(),'parameters':582888,'initial_shared_sha256':'init',
                'training_epochs':30,'inference_precision':'explicit_FP32_TF32_OFF','parity_gates':cfg['benchmark'],
                'gpu':'fixture','torch':'fixture','cuda':'fixture','build':{'original':3,'fast':5},
                'training_contract':{'config':cfg,'data':{'split':'fixture'}},'checkpoint_sha256':'old',
                'timing':{'trt_original':timing,'trt_fast':timing},
                'timing_repeat_reverse':{'trt_fast':timing},'per_image':images,
                'metrics':{'trt_original':{'all':score},'trt_fast':{'all':score}},
                'boundary':{k:{'bands':{'3':{'rmse_m':1.68}}} for k in ('trt_original','trt_fast')}}
            fresh=copy.deepcopy(report);fresh.update(variant=E.VARIANTS[0],checkpoint_sha256='new')
            refpath=root/'preflight_N1/trt_benchmark.json';newpath=root/E.VARIANTS[0]/'trt_benchmark.json'
            refpath.write_text(json.dumps(report));newpath.write_text(json.dumps(fresh))
            proof={'initial_shared_sha256':'init','checkpoint_sha256':'old'}
            with patch.object(H,'audit_history',return_value=proof),patch('builtins.print'):
                choice=compare(cfg);self.assertEqual(choice['selected_variant'],E.VARIANTS[0])
                self.assertFalse(choice['full_KITTI_baseline_promoted'])
                fresh['benchmark_session_id']='stale';newpath.write_text(json.dumps(fresh))
                with self.assertRaises(RuntimeError):compare(cfg)
                fresh['benchmark_session_id']='session'
                fresh['timing_repeat_reverse']['trt_fast']['synchronized_wall']['mean_ms']=4.1
                newpath.write_text(json.dumps(fresh))
                self.assertIsNone(compare(cfg)['selected_variant'])

    def test_preflight_and_final_use_original_reference_and_do_not_retrain_parent(self):
        import benchmark_fast as B
        source=(E.ROOT/'benchmark_fast.py').read_text()
        self.assertIn("parity(pred,predictions['original_fp32']",source)
        self.assertIn("('original',original,3),('fast',fast,5)",source)
        self.assertIn('timing_repeat_reverse',source)
        self.assertNotIn("len(ckpt['state']['rows'])!=30",source)
        import run
        with self.assertRaises(ValueError):run.train(CFG,E.REFERENCE_VARIANT)
        for name in ('preflight_N1/trt_benchmark.json','benchmark_session_id','checkpoint_sha256'):
            self.assertIn(name,source)

    def test_export_after_backward_keeps_weights(self):
        model=active(E.make_model(CFG,E.VARIANTS[0]).eval())
        model(*args())['D_full'].mean().backward()
        before=H.full_weight_hash(model);module=typed_deployment(model)
        with tempfile.TemporaryDirectory() as folder:
            export_onnx(module,dict(zip(E.INPUTS,args())),Path(folder)/'fast.onnx',22)
        self.assertEqual(before,H.full_weight_hash(model))
        self.assertTrue(all(not m.training for m in module.modules()))

    def test_notebook_resume_and_preflight_before_one_train30(self):
        nb=json.loads((E.ROOT/'Train_N1Fast_Fresh30_A100.ipynb').read_text())
        source='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
        self.assertEqual(source.count("command('train',VARIANT)"),1)
        self.assertIn("BUNDLE=frozen",source);self.assertIn('preflight[\'targets\'][\'all_passed\']',source)
        self.assertLess(source.index("command('preflight')"),source.index("command('train',VARIANT)"))
        self.assertIn("if not (RUN/VARIANT/'last.pth').is_file()",source)
        for cell in nb['cells']:
            if cell['cell_type']=='code':ast.parse(''.join(cell['source']))

    def test_syntax(self):
        for path in E.ROOT.rglob('*.py'):
            if '__pycache__' not in path.parts:ast.parse(path.read_text(encoding='utf-8'),filename=str(path))


if __name__=='__main__':unittest.main()
