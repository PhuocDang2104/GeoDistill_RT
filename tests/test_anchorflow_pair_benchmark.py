"""Coordinator QA. Tiny processes and fabricated reports are NOT GPU benchmarks."""
import ast
import contextlib
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import anchorflow_pair_benchmark as b

FOLDERS={'V10_1':ROOT/'drive_upload/AnchorFlow_v10_1_LiteMetric',
         'V11':ROOT/'drive_upload/AnchorFlow_v11_NODE'}

def configs():return {name:b.read_json(p/'config.json') for name,p in FOLDERS.items()}

def bundle(folder):
    folder.mkdir()
    (folder/'model.py').write_text('VALUE=1\n',encoding='utf-8')
    (folder/'demo.ipynb').write_text('mutable UI',encoding='utf-8')
    b.atomic_json(folder/'bundle_manifest.json',{'files':{name:b.digest(folder/name)
                   for name in ('model.py','demo.ipynb')}})
    return folder

def command(root,label,delay=.2,exit_code=0):
    # No files shared between these independent children except stdout queues.
    source=f"import time; print('hello {label}',flush=True); time.sleep({delay}); print('done {label}',flush=True); raise SystemExit({exit_code})"
    return b.Command(label,'qa',[sys.executable,'-u','-c',source],root,
                     root/f'{label}.log',root/'drive'/f'{label}.log')

def metric(value,pixels=25424992):
    return {'rmse_m':value,'mae_m':value/2,'irmse_km_inv':3.,'imae_km_inv':1.5,
            'abs_rel':.02,'delta1':.99,'pixels':pixels,'pred_below_0_1':0,
            'error_tail':{'5':{'pixels':5,'pixel_fraction':.001,'sse_fraction':.1,'sse_m2':1.,'mse_contribution_m2':.001}}}

def fabricated_reports(root):
    """Test parser schemas ONLY; all numbers here are invented."""
    jobs=[];cfgs=configs()
    for label,rmse in (('V10_1',1.),('V11',.95)):
        path=root/label;path.mkdir()
        (path/'best.pth').write_bytes(b'not a real checkpoint, parser QA only')
        provenance={'completed':True,'checkpoint_sha256':b.digest(path/'best.pth'),
                    'resolved_config_sha256':'dummy-config','bundle_manifest_sha256':'dummy-source',
                    'GPU_visible_token':'0'}
        for name in ('evaluate','profile','solver_audit'):
            b.atomic_json(path/f'{name}_artifact_record.json',provenance)
        score={key:metric(rmse) for key in ('all','0-20','20-40','40-60','60-80','80-120','edge')}
        report={'checkpoint_epoch':1,'checkpoint_selection':'rmse','samples':400,'teacher_at_inference':False,
                'final':score,'pre_anchor':score,'legacy_hard_anchor':score,
                'gt_boundary':{'bands':{'3':{'rmse_m':rmse,'pixels':50,'mae_m':.4}}},
                'stage_native_gt_metrics':{'D4':{'rmse_m':rmse+.1,'mae_m':.4,'pixels':100}}}
        b.atomic_json(path/'val_metrics.json',report)
        for selection in ('inverse','joint'):
            (path/f'best_{selection}.pth').write_bytes((path/'best.pth').read_bytes())
            b.atomic_json(path/f'evaluate_{selection}_artifact_record.json',provenance)
            b.atomic_json(path/f'val_metrics_{selection}.json',{**report,'checkpoint_epoch':0,'checkpoint_selection':selection})
        b.atomic_json(path/'training_status.json',{'status':'early_stopped','epochs_completed':3 if label=='V11' else 2})
        fields=['epoch','global_step','epoch_seconds','train_seconds','val_rmse_m','val_irmse_km_inv','val_native_D4_rmse_m']
        with (path/'train_log.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for index in range(3 if label=='V11' else 2):
                writer.writerow(dict(zip(fields,(index,index+1,12,10,rmse+(.1 if index==0 else 0),3.,1.1))))
        profile={'device':'MOCK GPU, NOT BENCHMARK','torch':'qa','precision':'bf16','channels_last':True,'runs':100,
                 'shape':[352,1216],'batch':1,'compiled_total':False,'untrained_weights':False,
                 'total_parameters':10,'total_conv_linear_macs':1000,'peak_cuda_allocated_mib':1.,
                 'runtime_backend':dict.fromkeys(('cudnn_enabled','cudnn_benchmark','cudnn_deterministic',
                    'cudnn_allow_tf32','matmul_allow_tf32','cuda_build','cudnn_version'),'mock'),
                 'real_scene_profile':{'samples':100,'indices':'i*13 mod400','wall_median_ms':2.,'wall_p95_ms':3.,
                                       'executed_nfe':2,'solver':{'nfe_mean':13}}}
        b.atomic_json(path/'profile.json',profile)
        if label=='V11':
            b.atomic_json(path/'profile_fixed_midpoint8.json',profile)
            b.atomic_json(path/'val_metrics_fixed_midpoint8.json',report)
            b.atomic_json(path/'solver_comparison.json',[{'solver':'parser_fixture','epoch':1,'rmse_m':rmse}])
        jobs.append({'label':label,'run_dir':str(path),'config':cfgs[label]})
    return jobs

class BundleContracts(unittest.TestCase):
    def test_both_real_bundles_still_sealed_and_shared_sources_match(self):
        for folder in FOLDERS.values():b.verify_bundle(folder)
        for name in b.SHARED_SOURCES:self.assertEqual((FOLDERS['V10_1']/name).read_bytes(),(FOLDERS['V11']/name).read_bytes())

    def test_mutable_ui_only_is_skipped(self):
        with tempfile.TemporaryDirectory() as temp:
            p=bundle(Path(temp)/'source');(p/'demo.ipynb').write_text('Colab saves outputs')
            destination=Path(temp)/'local';b.stage_bundle(p,destination)
            self.assertFalse((destination/'demo.ipynb').exists())
            (p/'model.py').write_text('changed code')
            with self.assertRaisesRegex(RuntimeError,'Bundle changed'):b.verify_bundle(p)

    def test_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=bundle(Path(temp)/'source')
            for name in ('../model.py','..\\model.py','/model.py'):
                b.atomic_json(p/'bundle_manifest.json',{'files':{name:'dummy'}})
                with self.assertRaises(ValueError):b.verify_bundle(p)

    def test_freeze_preflight_preserves_original(self):
        with tempfile.TemporaryDirectory() as temp:
            p=bundle(Path(temp)/'source');snapshot=Path(temp)/'frozen'
            b.freeze_snapshot(snapshot,p,{'epochs':40})
            before=(snapshot/'resolved_config.json').read_bytes()
            b.freeze_snapshot(snapshot,p,{'epochs':40})
            with self.assertRaisesRegex(RuntimeError,'Frozen recipe changed'):
                b.freeze_snapshot(snapshot,p,{'epochs':20})
            self.assertEqual(before,(snapshot/'resolved_config.json').read_bytes())

    def test_matching_and_fresh_contract(self):
        cfg=configs();self.assertTrue(b.matched_recipe(cfg)['matched_non_architecture_settings'])
        for key,value in (('batch_size',2),('epochs',20),('init_checkpoint','old.pth'),('learning_rate',.001)):
            changed=copy.deepcopy(cfg);changed['V11'][key]=value
            with self.assertRaises((ValueError,RuntimeError)):b.matched_recipe(changed)
        cfg['V10_1']['learned_step_size']=False
        with self.assertRaisesRegex(ValueError,'CURRENT'):b.matched_recipe(cfg)

    def test_same_invalid_recipe_is_rejected(self):
        for key,value in (('amp','fp16'),('flow_steps',3),('encoder_pretrained',False),('early_stopping',False)):
            cfg=configs()
            for c in cfg.values():c[key]=value
            with self.assertRaises(ValueError):b.matched_recipe(cfg)

    def test_visible_gpu_mapping(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-alpha,3'}):
            self.assertEqual(b.visible_gpu_token(0),'GPU-alpha');self.assertEqual(b.visible_gpu_token(1),'3')
            with self.assertRaises(ValueError):b.visible_gpu_token(2)
        self.assertEqual(b.process_environment('GPU-alpha','code')['CUDA_VISIBLE_DEVICES'],'GPU-alpha')

    def test_data_contract_checks_content_and_no_val_teacher(self):
        cfg=configs()['V11']
        report={'train':1600,'val':400,'test':1000,'manifest_sha256':cfg['expected_subset_sha256'],
                'size':[352,1216],'depth_scale':256,'validation_teacher_used':False,'relative_val_used':False,
                'teacher_report':{'cached_train':1600,'cached_val':0,
                    'selected_teacher_content_sha256':'06bd17a84faa5020692873ee598498486d2030f588b7429321fa0a70d9ebb7a6'},
                'relative_teacher':{'cached_train':1600,'cached_val':0,
                    'selected_content_sha256':'dd3b49a4474560fedbfc78504dd584c0234d790c65a356c2e74b6bb82759e86a'}}
        b.verify_data_contract(report,cfg)
        for key,value in (('val',399),('validation_teacher_used',True),('depth_scale',1000),('relative_val_used',True)):
            changed=copy.deepcopy(report);changed[key]=value
            with self.assertRaises(RuntimeError):b.verify_data_contract(changed,cfg)
        report['relative_teacher']['selected_content_sha256']='different'
        with self.assertRaisesRegex(RuntimeError,'Relative teacher content'):b.verify_data_contract(report,cfg)

class ProcessContracts(unittest.TestCase):
    def invoke(self,*args,**kwargs):
        with contextlib.redirect_stdout(io.StringIO()):return b.run_processes(*args,**kwargs)

    def test_parallel_overlap_and_separate_mirrored_logs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);commands=[command(root,'a',.6),command(root,'b',.6)]
            results=self.invoke(commands,True,root/'status.json',mirror_seconds=.05)
            self.assertLess(max(r['started_unix'] for r in results),min(r['finished_unix'] for r in results))
            for c in commands:
                self.assertEqual(c.local_log.read_text(),c.drive_log.read_text())
                self.assertIn(f'done {c.label}',c.drive_log.read_text())
            self.assertEqual(b.read_json(root/'status.json')['status'],'completed')

    def test_sequential_has_no_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            results=self.invoke([command(root,'a'),command(root,'b')],False,root/'status.json')
            self.assertGreaterEqual(results[1]['started_unix'],results[0]['finished_unix'])

    def test_failed_peer_survives_and_failure_record_keeps_both(self):
        for parallel in (True,False):
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                with self.assertRaises(b.ProcessGroupError) as caught:
                    self.invoke([command(root,'fail',.05,7),command(root,'peer',.4)],parallel,root/'status.json')
                self.assertEqual(len(caught.exception.results),2)
                self.assertIn('done peer',(root/'drive/peer.log').read_text())
                report=b.read_json(root/'status.json');self.assertEqual(report['status'],'failed')
                self.assertEqual(len(report['results']),2)

    def test_launch_failure_cleans_up_launched_children(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);bad=command(root,'bad');bad.args=['nonexistent-anchorflow-python-command']
            real_popen=b.subprocess.Popen;spawned=[]
            def tracked(*args,**kwargs):
                p=real_popen(*args,**kwargs);spawned.append(p);return p
            with patch.object(b.subprocess,'Popen',side_effect=tracked):
                with self.assertRaises(OSError):self.invoke([command(root,'peer',10),bad],True,root/'status.json')
            self.assertTrue(all(p.poll() is not None for p in spawned))
            self.assertEqual(b.read_json(root/'status.json')['status'],'orchestration_failed')

    def test_evaluation_artifact_binds_correct_selection_and_gpu(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);code=root/'code';code.mkdir();run_dir=root/'run';run_dir.mkdir()
            (run_dir/'best_inverse.pth').write_bytes(b'fixture-selected-checkpoint')
            b.atomic_json(code/'resolved_config.json',{'mode':'parser QA'})
            b.atomic_json(code/'bundle_manifest.json',{'mode':'parser QA'})
            job={'label':'V11','code':str(code),'run_dir':str(run_dir),'variant':'dual_teacher','gpu':'1'}
            with patch.object(b,'run_processes',return_value=[]):
                b.run_stage([job],'evaluate',gpu='0',extra=('--checkpoint-selection','inverse'),suffix='_inverse')
            record=b.read_json(run_dir/'evaluate_inverse_artifact_record.json')
            self.assertEqual(record['checkpoint'],'best_inverse.pth')
            self.assertEqual(record['selection'],'inverse');self.assertEqual(record['GPU_visible_token'],'0')

    def test_checkpoint_change_during_profile_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);code=root/'code';code.mkdir();run_dir=root/'run';run_dir.mkdir()
            checkpoint=run_dir/'best.pth';checkpoint.write_bytes(b'before')
            b.atomic_json(code/'resolved_config.json',{});b.atomic_json(code/'bundle_manifest.json',{})
            job={'label':'V11','code':str(code),'run_dir':str(run_dir),'variant':'dual_teacher','gpu':'0'}
            def changed(*args,**kwargs):checkpoint.write_bytes(b'after');return []
            with patch.object(b,'run_processes',side_effect=changed):
                with self.assertRaisesRegex(RuntimeError,'Checkpoint changed during profile'):b.run_stage([job],'profile')

class ReportContracts(unittest.TestCase):
    def test_actual_config_cell_isolates_new_tags_and_freezes_resume(self):
        # Execute the notebook's actual config cell with a fake CUDA inventory.
        # This verifies paths/freezing, not CUDA kernels or hardware performance.
        notebook=json.loads((ROOT/'notebooks/AnchorFlow_Compare_V10_1_V11_Fresh40.ipynb').read_text(encoding='utf-8'))
        cell=''.join(notebook['cells'][7]['source'])
        cuda=SimpleNamespace(device_count=lambda:2,device=lambda i:contextlib.nullcontext(),
            mem_get_info=lambda:(64*2**30,80*2**30),get_device_capability=lambda i:(9,0),
            is_bf16_supported=lambda **kw:True,get_device_name=lambda i:'FAKE CUDA for config-cell QA')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);codes={name:root/name for name in FOLDERS}
            for name in FOLDERS:b.stage_bundle(FOLDERS[name],codes[name])
            ns={'torch':SimpleNamespace(cuda=cuda,__version__='mock-config-test'),'bench':b,'time':time,
                'json':json,'shutil':b.shutil,'Path':Path,'LOCAL_ROOT':root,'DRIVE_RUNS':root/'drive-runs',
                'DRIVE_DATA':root/'data','CODES':codes,'TRAIN_GPU_INDICES':None,'PROFILE_GPU_INDEX':0,
                'AMP_REQUEST':'auto','PARALLEL_TRAIN':True,'SHARED_GPU_MIN_FREE_GIB':20,'BENCH_TAG':'first',
                'BATCH_SIZE':4,'ACCUMULATION':1,'WORKERS':2,'CPU_THREADS_PER_MODEL':4,
                'runner_file':ROOT/'scripts/anchorflow_pair_benchmark.py','RUNTIME_SHA256':'config-cell QA'}
            with contextlib.redirect_stdout(io.StringIO()),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-test-a,GPU-test-b'}):
                exec(cell,ns)
                first=ns['WORK'];original_root=ns['COMPARE_ROOT']
                first_snapshot=(original_root/'source_bundle/V11/resolved_config.json').read_bytes()
                ns['BENCH_TAG']='second';exec(cell,ns)
                second=ns['WORK'];self.assertNotEqual(first,second)
                self.assertEqual(ns['JOBS'][0]['config']['work'],ns['JOBS'][1]['config']['work'])
                self.assertEqual([j['gpu'] for j in ns['JOBS']],['GPU-test-a','GPU-test-b'])
                exec(cell,ns) # Exact unchanged recipe resumes without nesting WORK again.
                self.assertEqual(second,ns['WORK'])
                ns['PARALLEL_TRAIN']=False;exec(cell,ns) # Scheduling only can change.
                self.assertEqual(second,ns['WORK'])
                ns['BATCH_SIZE']=2
                with self.assertRaisesRegex(RuntimeError,'Frozen recipe changed'):exec(cell,ns)
            self.assertEqual(first_snapshot,(original_root/'source_bundle/V11/resolved_config.json').read_bytes())

    def test_report_parser_and_same_epoch_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);jobs=fabricated_reports(root)
            summary=b.compare_reports(jobs,root/'comparison')
            self.assertAlmostEqual(summary['deltas']['V11_minus_V10_1_RMSE_m'],-.05)
            self.assertEqual(len(summary['efficiency']),3)
            self.assertEqual(summary['common_epoch_budget'][0]['common_completed_epochs'],2)
            self.assertEqual([r['epoch'] for r in summary['selections']],[1,0,0,1,0,0])
            self.assertTrue((root/'comparison/comparison_dashboard.png').is_file())
            self.assertTrue((root/'comparison/comparison_tails.csv').is_file())

    def test_gpu_mismatch_is_not_silently_compared(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);jobs=fabricated_reports(root)
            p=root/'V11/profile.json';r=b.read_json(p);r['device']='another GPU';b.atomic_json(p,r)
            with self.assertRaisesRegex(RuntimeError,'mismatch'):b.compare_reports(jobs,root/'comparison')

    def test_accuracy_profile_checkpoint_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);jobs=fabricated_reports(root)
            p=root/'V11/profile_artifact_record.json';r=b.read_json(p);r['checkpoint_sha256']='another';b.atomic_json(p,r)
            with self.assertRaisesRegex(RuntimeError,'provenance mismatch'):b.compare_reports(jobs,root/'comparison')

    def test_static_solver_epoch_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);jobs=fabricated_reports(root)
            p=root/'V11/val_metrics_fixed_midpoint8.json';r=b.read_json(p);r['checkpoint_epoch']=2;b.atomic_json(p,r)
            with self.assertRaisesRegex(RuntimeError,'different checkpoint'):b.compare_reports(jobs,root/'comparison')

    def test_changed_pixel_support_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);jobs=fabricated_reports(root)
            p=root/'V11/val_metrics.json';r=b.read_json(p);r['final']['all']['pixels']=5;b.atomic_json(p,r)
            with self.assertRaisesRegex(RuntimeError,'GT support changed'):b.compare_reports(jobs,root/'comparison')

    def test_notebook_cells_and_embedded_helper(self):
        notebook=json.loads((ROOT/'notebooks/AnchorFlow_Compare_V10_1_V11_Fresh40.ipynb').read_text(encoding='utf-8'))
        embedded=None
        for cell in notebook['cells']:
            source=''.join(cell['source'])
            if cell['cell_type']=='code':
                tree=ast.parse(source)
                if source.startswith('RUNTIME_SOURCE = '):
                    embedded={};exec(compile(ast.Module(body=tree.body[:2],type_ignores=[]),'embedded','exec'),embedded)
        self.assertIsNotNone(embedded)
        source=(ROOT/'scripts/anchorflow_pair_benchmark.py').read_text(encoding='utf-8')
        self.assertEqual(embedded['RUNTIME_SOURCE'],source)
        self.assertEqual(embedded['RUNTIME_SHA256'],hashlib.sha256(source.encode()).hexdigest())

if __name__=='__main__':unittest.main()
