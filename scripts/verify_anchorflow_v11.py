"""CPU verification, real KITTI inputs, synthetic teacher targets, no GPU claims."""
import gc
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
FOLDER=ROOT/'drive_upload/AnchorFlow_v11_NODE'
sys.path.insert(0,str(FOLDER))
import run
from model import Deploy
from losses import objective
from data import KITTIDataset,write_json

def main():
    torch.set_num_threads(4)
    cfg=json.loads((FOLDER/'config.json').read_text())
    tested=subprocess.run([sys.executable,'-X','utf8','-m','unittest','discover','-s',str(FOLDER),'-v'],
                          capture_output=True,text=True,encoding='utf-8')
    print(tested.stdout+tested.stderr,flush=True);assert tested.returncode==0
    proof={'tests':32,'tests_returncode':0,'device':'CPU','gpu_available':torch.cuda.is_available(),
           'trained_v11_accuracy_measured':False,'gpu_training_or_latency_measured':False}
    with tempfile.TemporaryDirectory(prefix='anchorflow_v11_stage_') as temp:
        for p in FOLDER.iterdir():
            if p.is_file():shutil.copy2(p,Path(temp)/p.name)
        staged=subprocess.run([sys.executable,'-X','utf8','-m','unittest','discover','-s',temp,'-v'],
            cwd=temp,capture_output=True,text=True,encoding='utf-8')
        assert staged.returncode==0,staged.stdout+staged.stderr
        proof['isolated_staged_tests_returncode']=0
    ds=object.__new__(KITTIDataset);ds.root=ROOT/'data/teacher_subset_2000/kitti_bundle'
    ds.split,ds.teacher='val',False
    ds.rows=[r.split() for r in (ds.root/'splits/val_400.txt').read_text().splitlines()]
    def batch(index):return {k:v[None] if torch.is_tensor(v) else v for k,v in ds[index].items()}
    proof['real_kitti_full_resolution_backward']={}
    for precision in ('fp32','bf16'):
        torch.manual_seed(42);model=run.make_model(cfg,torch.device('cpu'),False).train();model.freeze_encoder_bn()
        b=batch(0);b['teacher']=b['gt']+10;b['confidence']=torch.ones_like(b['gt'])
        b['relative']=b['rgb'][:,:1]+.05*b['rgb'][:,1:2];b['relative_confidence']=torch.ones_like(b['gt'])
        with torch.autocast('cpu',dtype=torch.bfloat16,enabled=precision=='bf16'):
            out=model(*(b[k] for k in ('rgb','sparse','mask','K')))
        loss,stats=objective(out,b,b['mask']*0,4,cfg);assert torch.isfinite(loss)
        loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        grad=model.dynamics.reaction.weight.grad
        assert grad is not None and grad.abs().sum()>0
        proof['real_kitti_full_resolution_backward'][precision]={'finite':True,'loss':float(loss.detach()),
            'reaction_gradient_l1':float(grad.abs().sum()),'solver':model.dynamics.last_solver_report,
            'teacher_targets':'SYNTHETIC; actual Drive-cache smoke is in Colab notebook',
            'rgb_sparse_gt_K':'real KITTI val0,352x1216','pretrained_weights_loaded':False}
        print('FULL BACKWARD',precision,proof['real_kitti_full_resolution_backward'][precision],flush=True)
        del model,out,loss,b;gc.collect()
    torch.manual_seed(123);model=run.make_model(cfg,torch.device('cpu'),False).eval()
    model.set_diagnostics(False);x=run.sample_inputs(torch.device('cpu'),channels_last=False)
    with torch.inference_mode():proof['adaptive_complexity']=run.count_operations(model,x)
    proof['adaptive_synthetic_solver']=dict(model.dynamics.last_solver_report)
    model.dynamics.method='midpoint'
    with torch.inference_mode():proof['fixed_midpoint8_complexity']=run.count_operations(model,x)
    # Open readout heads so ONNX is not checked only on a trivial flat output.
    with torch.no_grad():
        model.phase_context[-1].weight.normal_(0,.01)
        model.phase2.delta.weight.normal_(0,.01)
        model.phase2.metric[-1].weight.normal_(0,.01)
        model.detail1.delta.weight.normal_(0,.01)
        model.dynamics.reaction.weight.normal_(0,.02)
    import onnx
    import onnxruntime as ort
    deploy=Deploy(model).eval()
    with tempfile.TemporaryDirectory(prefix='anchorflow_v11_onnx_') as temp:
        path=Path(temp)/'fixed_midpoint8.onnx'
        torch.onnx.export(deploy,x,str(path),input_names=['rgb','sparse','mask','K'],output_names=['depth_m'],
                          opset_version=17,dynamo=False)
        onnx.checker.check_model(onnx.load(path))
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.log_severity_level=3
        session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        cases=[]
        for seed in (0,7,42):
            torch.manual_seed(seed);sample=run.sample_inputs(torch.device('cpu'),channels_last=False)
            cases.append((f'random_{seed}',sample))
            cases.append((f'empty_{seed}',(sample[0],sample[1]*0,sample[2]*0,sample[3])))
        for index in (0,199,399):
            b=batch(index);cases.append((f'real_{index}',tuple(b[k] for k in ('rgb','sparse','mask','K'))))
        errors={}
        with torch.inference_mode():
            for label,sample in cases:
                expected=deploy(*sample).numpy()
                actual=session.run(None,{k:v.numpy() for k,v in zip(('rgb','sparse','mask','K'),sample)})[0]
                err=float(np.abs(expected-actual).max());errors[label]=err
                assert np.isfinite(actual).all() and np.allclose(actual,expected,atol=.01,rtol=1e-4),(label,err)
                print('ONNX',label,err,flush=True)
        proof['onnx']={'checker':True,'onnxruntime_cpu_parity':True,'cases':errors,
            'solver':'fixed midpoint8','adaptive_solver_exported':False,'several_zero_init_heads_opened':True,
            'tensorrt_engine_built':False,'target_device_latency_measured':False}
    proof['source_sha256']=run.source_hash()
    proof['verified_files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in FOLDER.iterdir()
        if p.is_file() and p.suffix in ('.py','.json') and p.name not in ('local_verification.json','bundle_manifest.json')}
    write_json(FOLDER/'local_verification.json',proof)
    print('VERIFIED',proof['adaptive_complexity']['total_parameters'],proof['adaptive_complexity']['total_conv_linear_macs'],flush=True)

if __name__=='__main__':main()
