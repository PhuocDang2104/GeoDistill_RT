"""CPU proof: isolated tests, real full-image backward, pruned-init val, ONNX.

No claimed GPU training metric/latency. Synthetic teachers in local backward
are explicitly marked; production notebook smoke uses the real Drive cache.
"""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import torch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'drive_upload/AnchorFlow_v10_1_LiteMetric'
AUDIT=ROOT/'results/anchorflow_v10_completed_audit'
sys.path.insert(0,str(FOLDER))
import run
from model import AnchorFlowEdge,Deploy
from losses import objective
from data import KITTIDataset,write_json
from metrics import Metrics

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    torch.set_num_threads(4)
    config=json.loads((FOLDER/'config.json').read_text())
    tested=subprocess.run([sys.executable,'-X','utf8','-m','unittest','discover','-s',str(FOLDER),'-v'],capture_output=True,text=True,encoding='utf-8')
    print(tested.stdout+tested.stderr,flush=True);assert tested.returncode==0
    proof={'tests':30,'tests_returncode':tested.returncode,'device':'CPU','gpu_available':False,'trained_v10_1_accuracy_measured':False}
    with tempfile.TemporaryDirectory(prefix='anchorflow_v10_1_stage_') as temp:
        for p in FOLDER.iterdir():
            if p.is_file():shutil.copy2(p,Path(temp)/p.name)
        staged=subprocess.run([sys.executable,'-X','utf8','-m','unittest','discover','-s',temp,'-v'],cwd=temp,capture_output=True,text=True,encoding='utf-8')
        assert staged.returncode==0,staged.stdout+staged.stderr
        proof['isolated_staged_tests_returncode']=staged.returncode
    ds=object.__new__(KITTIDataset);ds.root=ROOT/'data/teacher_subset_2000/kitti_bundle';ds.split,ds.teacher='val',False
    ds.rows=[r.split() for r in (ds.root/'splits/val_400.txt').read_text().splitlines()]
    def b(index):return {k:v[None] if torch.is_tensor(v) else v for k,v in ds[index].items()}
    parent=torch.load(AUDIT/'best.pth',map_location='cpu',weights_only=False)
    model=AnchorFlowEdge().eval();loaded=model.load_state_dict(parent['model'],strict=False)
    assert all(n.startswith(('phase_context.','dynamics.step_head.')) for n in loaded.missing_keys)
    with torch.no_grad():model.dynamics.step_head.bias.fill_(4.) # optional warm migration init
    assert all(n.startswith('dynamics.controller.') for n in loaded.unexpected_keys)
    proof['migration_keys']={'new':loaded.missing_keys,'removed':loaded.unexpected_keys}
    initial=Metrics(torch.device('cpu'))
    model.set_diagnostics(False)
    with torch.inference_mode():
        for index in range(400):
            sample=b(index);out=model(*(sample[k] for k in ('rgb','sparse','mask','K')))
            initial.update(out['D_full'],sample['gt'],sample['gt_mask'],sample['rgb'])
            if (index+1)%100==0:print(f'Fixed2 model-only initial validation {index+1}/400',flush=True)
    report=initial.report();assert report['all']['pixels']==25424992
    proof['pruned_parent_initial_validation_cpu_fp32']=report
    write_json(AUDIT/'v10_1_pruned_initial_val_metrics_cpu.json',report)
    # Local backwards: real KITTI inputs, synthetic privileged targets, not a
    # real-teacher accuracy measurement and not student training.
    proof['real_kitti_backward']={}
    for precision in ('fp32','bf16'):
        sample=b(0);sample['teacher']=sample['gt']+10;sample['confidence']=torch.ones_like(sample['gt'])
        sample['relative']=sample['rgb'][:,:1]+.05*sample['rgb'][:,1:2];sample['relative_confidence']=torch.ones_like(sample['gt'])
        tested_model=AnchorFlowEdge().train();tested_model.load_state_dict(parent['model'],strict=False);tested_model.freeze_encoder_bn()
        holdout=sample['mask']*0
        with torch.autocast('cpu',dtype=torch.bfloat16,enabled=precision=='bf16'):
            pred=tested_model(*(sample[k] for k in ('rgb','sparse','mask','K')))
        loss,stats=objective(pred,sample,holdout,4,config);assert torch.isfinite(loss)
        loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in tested_model.parameters())
        new=tested_model.phase_context[-1].weight.grad
        assert new is not None and new.abs().sum()>0
        proof['real_kitti_backward'][precision]={'finite':True,'loss':float(loss.detach()),
           'new_context_gradient_l1':float(new.abs().sum()),'real_drive_teacher_cache_used':False,
           'teacher_targets':'synthetic; real RGB/sparse/GT/intrinsics; production notebook has real-cache smoke'}
        del tested_model,pred,loss,sample
    inputs=run.sample_inputs(torch.device('cpu'),channels_last=False)
    model.set_diagnostics(False)
    proof['complexity']=run.count_operations(model,inputs)
    # Open the added branch for meaningful parity, NOT an accuracy improvement.
    torch.manual_seed(123)
    with torch.no_grad():
        model.phase_context[-1].weight.normal_(0,.01);model.phase_context[-1].bias.normal_(0,.01)
    import onnx
    import onnxruntime as ort
    deploy=Deploy(model).eval();path=AUDIT/'v10_1_open_branch_parity.onnx'
    torch.onnx.export(deploy,inputs,str(path),input_names=['rgb','sparse','mask','K'],output_names=['depth_m'],opset_version=17,dynamo=False)
    onnx.checker.check_model(onnx.load(path))
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.log_severity_level=3
    session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    cases=[]
    for seed in (0,7,42):
        torch.manual_seed(seed);x=run.sample_inputs(torch.device('cpu'),channels_last=False)
        cases.append((f'random_{seed}',x));cases.append((f'empty_{seed}',(x[0],x[1]*0,x[2]*0,x[3])))
    for index in (0,199,399):
        sample=b(index);cases.append((f'real_{index}',tuple(sample[k] for k in ('rgb','sparse','mask','K'))))
    errors={}
    with torch.inference_mode():
        for label,x in cases:
            expected=deploy(*x).numpy();actual=session.run(None,{k:v.numpy() for k,v in zip(('rgb','sparse','mask','K'),x)})[0]
            error=float(np.max(np.abs(expected-actual)));errors[label]=error
            assert np.isfinite(actual).all() and np.allclose(actual,expected,atol=.01,rtol=1e-4),(label,error)
            print('ONNX parity:',label,error,flush=True)
    proof['onnx']={'checker':True,'onnxruntime_cpu_parity':True,'cases':errors,'new_branch_opened_for_test':True,
       'trained_parent_weights':True,'tensorrt_engine_built':False,'target_device_runtime_measured':False}
    proof['source_sha256']=run.source_hash()
    proof['verified_files']={p.name:sha(p) for p in FOLDER.iterdir() if p.is_file() and p.suffix in ('.py','.json') and p.name not in ('local_verification.json','bundle_manifest.json')}
    write_json(FOLDER/'local_verification.json',proof)
    print('VERIFIED:',proof['complexity']['total_parameters'],proof['complexity']['total_conv_linear_macs'],flush=True)

if __name__=='__main__':main()
