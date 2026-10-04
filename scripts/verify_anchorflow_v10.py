"""Executable proof: contracts, real KITTI gradient fixtures, actual trained-block ORT.

No fresh V10 accuracy or GPU inference claims. Diagnostic loaded V9.1 blocks are
never copied into the upload folder or used by the fresh training workflow.
"""
import gc
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
import onnx
import onnxruntime as ort

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'drive_upload/AnchorFlow_v10_AdaptiveJet'
AUDIT=ROOT/'results/anchorflow_v10_verification'
sys.path.insert(0,str(FOLDER))
import run
from data import KITTIDataset,write_json
from losses import objective

def dataset():
    ds=object.__new__(KITTIDataset)
    ds.root=ROOT/'data/teacher_subset_2000/kitti_bundle'
    ds.split,ds.teacher='val',False
    ds.rows=[r.split() for r in (ds.root/'splits/val_400.txt').read_text().splitlines()]
    return ds

def batch(ds,i): return {k:v[None] if torch.is_tensor(v) else v for k,v in ds[i].items()}

def trained_block_diagnostic(cfg):
    model=run.make_model(cfg,torch.device('cpu'),False).eval()
    checkpoint=torch.load(ROOT/'results/anchorflow_v9_1_completed_audit/best.pth',map_location='cpu',weights_only=False)
    loaded=model.load_state_dict(checkpoint['model'],strict=False)
    assert not loaded.unexpected_keys
    assert loaded.missing_keys and all(k.startswith('dynamics.controller.') for k in loaded.missing_keys)
    return model.to(memory_format=torch.contiguous_format)

def exports(cfg,ds):
    model=trained_block_diagnostic(cfg)
    # Nonconstant/time-conditioned policy: exit after THREE, not constant fresh bias.
    with torch.no_grad():
        for module in model.dynamics.controller.modules():
            if isinstance(module,nn.Linear): module.weight.zero_();module.bias.zero_()
        model.dynamics.controller.net[0].weight[0,9]=1
        model.dynamics.controller.net[2].weight[0,0]=1
        model.dynamics.controller.net[4].weight[1,0]=100
        model.dynamics.controller.net[4].bias.copy_(torch.tensor([4.,-35.]))
    keys=('D4_step1','D4_step2','D4_step3','D4_step4','D4','D2_query','D2','D1','D_full','adaptive_selected_nfe','adaptive_terminal_time')
    class Stages(nn.Module):
        def __init__(self,m): super().__init__();self.model=m
        def forward(self,rgb,sparse,mask,K):
            o=self.model(rgb,sparse,mask,K);return tuple(o[k] for k in keys)
    samples=[]
    for seed in (0,7,42):
        torch.manual_seed(seed); x=run.sample_inputs('cpu',channels_last=False)
        samples.append((f'seed{seed}_observed',x));r,s,m,K=x
        samples.append((f'seed{seed}_empty',(r,s*0,m*0,K)))
    for idx in (0,199,399):
        b=batch(ds,idx);samples.append((f'real_kitti_{idx}',tuple(b[k].contiguous() for k in ('rgb','sparse','mask','K'))))
    all_results={}
    for mode in ('masked_adaptive','static4'):
        model.dynamics.mode=mode
        wrapped=Stages(model).eval();path=AUDIT/(mode+'.onnx')
        torch.onnx.export(wrapped,samples[0][1],str(path),input_names=['rgb','sparse','mask','K'],
            output_names=list(keys),opset_version=17,dynamo=False)
        graph=onnx.load(path);onnx.checker.check_model(graph)
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.log_severity_level=3
        session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        cases={}
        for label,x in samples:
            with torch.no_grad(): expected=wrapped(*x)
            actual=session.run(None,{k:t.numpy() for k,t in zip(('rgb','sparse','mask','K'),x)})
            errors={}
            for k,a,e in zip(keys,actual,expected):
                err=float(np.max(np.abs(a-e.numpy())))
                assert np.isfinite(a).all() and np.allclose(a,e.numpy(),atol=.01,rtol=1e-4),(mode,label,k,err)
                errors[k]=err
            assert int(actual[-2][0])==(3 if mode=='masked_adaptive' else 4)
            cases[label]=errors
            print('ORT PASS',mode,label,max(errors.values()),flush=True)
        all_results[mode]={'cases':cases,'max_abs_m':max(max(v.values()) for v in cases.values()),
            'opset':17,'conditional_compute_exported':False,'onnxruntime_cpu_parity':True}
        del session,wrapped;gc.collect()
    return {'graphs':all_results,'loaded_reference':'actual V9.1 best local epoch1; diagnostic only',
            'controller':'time-dependent synthetic diagnostic, not trained V10',
            'student_weights_in_upload':False,'fresh_v10_accuracy_measured':False}

def main():
    torch.set_num_threads(4);AUDIT.mkdir(parents=True,exist_ok=True)
    tested=subprocess.run([sys.executable,'-X','utf8','-m','unittest','discover','-s',str(FOLDER),'-v'],
        cwd=FOLDER,capture_output=True,text=True,encoding='utf-8')
    log=tested.stdout+tested.stderr;print(log,flush=True);assert tested.returncode==0
    with tempfile.TemporaryDirectory(prefix='anchorflow_v10_staged_') as directory:
        dest=Path(directory)
        for p in FOLDER.iterdir():
            if p.is_file() and (p.suffix=='.py' or p.name=='config.json'):shutil.copy2(p,dest/p.name)
        staged=subprocess.run([sys.executable,'-X','utf8','-m','unittest','discover','-s',str(dest),'-v'],
            cwd=dest,capture_output=True,text=True,encoding='utf-8')
        if staged.returncode:print(staged.stdout+staged.stderr);raise RuntimeError('Colab-staged contracts failed')
    cfg=json.loads((FOLDER/'config.json').read_text());ds=dataset();gradients={}
    for precision in ('fp32','bf16'):
        run.seed_everything(42);model=run.make_model(cfg,torch.device('cpu'),False).train();model.freeze_encoder_bn()
        b=batch(ds,0)
        b.update(teacher=torch.full_like(b['gt'],20),confidence=torch.full_like(b['gt'],.8),
            relative=torch.linspace(-1,1,1216)[None,None,None].expand(1,1,352,1216).clone(),
            relative_confidence=torch.ones_like(b['gt']))
        with torch.autocast('cpu',dtype=torch.bfloat16,enabled=precision=='bf16'):
            o=model(*(b[k] for k in ('rgb','sparse','mask','K')))
        loss,stats=objective(o,b,b['mask']*0,4,cfg);loss.backward()
        assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        heads={k:float(p.grad.abs().sum()) for k,p in model.named_parameters() if k in (
            'dynamics.reaction.weight','dynamics.controller.net.4.bias','phase2.metric.4.weight')}
        assert all(v>0 for v in heads.values())
        gradients[precision]={'finite':True,'loss':float(loss.detach()),'head_gradient_l1':heads,
            'relative_pair_coverage':float(stats['relative_pair_coverage'])}
        print('Real KITTI backward PASS',precision,gradients[precision],flush=True)
        del model,b,o,loss;gc.collect()
    model=run.make_model(cfg,torch.device('cpu'),False).eval()
    complexity={}
    for policy in ('fixed3','fixed4','learned4','adaptive'):
        model.dynamics.policy=policy
        complexity[policy]=run.count_operations(model,run.sample_inputs('cpu'))
    del model;gc.collect()
    parity=exports(cfg,ds)
    proof={'tests':int(re.search(r'Ran (\d+) tests',log).group(1)),'tests_returncode':0,
        'colab_staged_tests_returncode':staged.returncode,'tests_log':log,'source_sha256':run.source_hash(),
        'python':sys.version,'torch':str(torch.__version__),'real_kitti_backward':gradients,
        'teacher_fixture':'Synthetic metric/relative targets; actual KITTI RGB/sparse/GT/K. No student checkpoint loaded for backward.',
        'complexity_by_policy':complexity,'onnx':parity,'gpu_training_tested':False,
        'gpu_latency_measured':False,'fresh_v10_training_accuracy_measured':False,
        'generator_unchanged':(FOLDER/'generate_relative.py').read_bytes()==(ROOT/'drive_upload/AnchorFlow_v9_Consensus/generate_relative.py').read_bytes()}
    verified=[p for p in FOLDER.iterdir() if p.is_file() and
        (p.suffix=='.py' or p.name in ('config.json','requirements.txt','teacher_requirements.txt') or p.name.endswith('_contract.json'))]
    proof['verified_files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in verified}
    write_json(FOLDER/'local_verification.json',proof)
    print('VERIFIED',complexity['adaptive']['total_parameters'],complexity['adaptive']['total_conv_linear_macs'],flush=True)

if __name__=='__main__':main()
