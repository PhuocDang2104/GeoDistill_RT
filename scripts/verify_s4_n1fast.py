"""Contracts + full352x1216 ORIGINAL-vs-FAST PyTorch/ONNX, isolated RAM workers."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'drive_upload/AnchorFlow_S4_UPIR_N1Fast_30ep_A100'


def export_reference(folder,kind):
    sys.path.insert(0,str(CODE))
    import numpy as np
    import onnx
    import torch
    import experiment as E
    from deployment import typed_deployment,export_onnx
    from test_contracts import active
    torch.set_num_threads(2);cfg=json.loads((CODE/'config.json').read_text())
    model=active(E.make_model(cfg,E.VARIANTS[0]).eval())
    original=typed_deployment(model,fast=False)
    module=typed_deployment(model,fast=kind=='fast')
    torch.manual_seed(824)
    rgb=torch.rand(1,3,352,1216);mask=(torch.rand(1,1,352,1216)>.95).float()
    sparse=(10+50*torch.rand_like(mask))*mask
    inputs=(rgb,sparse,mask,torch.tensor([[[700.,0.,607.],[0.,700.,175.],[0.,0.,1.]]]))
    with torch.no_grad():
        expected=original(*inputs)
        np.save(folder/'expected.npy',expected.numpy())
        actual=module(*inputs);error=(actual-expected).abs()
        parity={'mean_abs_m':float(error.mean()),'max_abs_m':float(error.max())}
        if parity['mean_abs_m']>.001 or parity['max_abs_m']>.06:raise RuntimeError(('Original PyTorch vs rewrite',parity))
        np.savez(folder/'inputs.npz',**{k:v.numpy() for k,v in zip(E.INPUTS,inputs)})
        export_onnx(module,dict(zip(E.INPUTS,inputs)),folder/'model.onnx',22)
    nodes={}
    for node in onnx.load(str(folder/'model.onnx')).graph.node:nodes[node.op_type]=nodes.get(node.op_type,0)+1
    (folder/'torch_parity.json').write_text(json.dumps({'vs_original_pytorch':parity,'onnx_nodes':nodes},indent=2))
    print(kind,'full352x1216 export PASS',parity,flush=True)


def ort_check(folder):
    import numpy as np
    import onnxruntime as ort
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    opts.enable_cpu_mem_arena=False;opts.enable_mem_pattern=False
    opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    session=ort.InferenceSession(str(folder/'model.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
    with np.load(folder/'inputs.npz') as data:actual=session.run(None,{k:data[k] for k in data.files})[0]
    expected=np.load(folder/'expected.npy');error=np.abs(actual-expected)
    result={'shape':list(actual.shape),'opset':22,'onnxruntime':ort.__version__,
            'mean_abs_m_vs_ORIGINAL_pytorch':float(error.mean()),
            'max_abs_m_vs_ORIGINAL_pytorch':float(error.max()),
            **json.loads((folder/'torch_parity.json').read_text())}
    if not np.isfinite(actual).all() or result['mean_abs_m_vs_ORIGINAL_pytorch']>.001 or result['max_abs_m_vs_ORIGINAL_pytorch']>.06:
        raise RuntimeError(result)
    (folder/'parity.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


def main():
    # Verify/repack frozen bytes only. Never rebuild notebook/policy/manifest.
    subprocess.run([sys.executable,'-B',str(ROOT/'scripts/package_s4_n1fast.py')],cwd=ROOT,check=True)
    test=subprocess.run([sys.executable,'-B','-X','utf8','-m','unittest','discover','-s',str(CODE),'-p','test_contracts.py','-v'],
                        cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    print(test.stdout+test.stderr,flush=True)
    if test.returncode:raise SystemExit(test.returncode)
    parities={}
    for kind in ('original','fast'):
        with tempfile.TemporaryDirectory(prefix='n1fast_onnx_') as tmp:
            for worker in ('export','ort'):
                subprocess.run([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--worker',worker,
                                '--folder',tmp,'--kind',kind],cwd=ROOT,check=True)
            parities[kind]=json.loads((Path(tmp)/'parity.json').read_text())
    report={'date':'2026-10-21','passed':True,'device':'CPU',
            'contract_tests':int(re.search(r'Ran (\d+) tests',test.stderr)[1]),
            'full_resolution_parity':parities,'comparison_reference':'ORIGINAL unoptimized FP32 PyTorch in both cases',
            'synthetic_probe':'Nonzero metric/PIR/uncertainty/Jet-context/fusion added lanes; not trained checkpoint/data metrics',
            'not_verified':['A100 training','TensorRT parser/build/capture','trained400 metrics','3.9ms target','Orin','full-KITTI generalization']}
    output=ROOT/'benchmarks/q1_n1fast_2026_10_21/local_reverification.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    subprocess.run([sys.executable,'-B',str(ROOT/'scripts/package_s4_n1fast.py')],cwd=ROOT,check=True)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=('export','ort'));p.add_argument('--kind',choices=('original','fast'));p.add_argument('--folder',type=Path)
    args=p.parse_args()
    if args.worker=='export':export_reference(args.folder,args.kind)
    elif args.worker=='ort':ort_check(args.folder)
    else:main()
