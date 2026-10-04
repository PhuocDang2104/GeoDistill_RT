"""Extra ONNX proof with genuinely nonzero, data-dependent timestep weights."""
import json
import sys
from pathlib import Path
import torch
import numpy as np
import onnx
import onnxruntime as ort

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'drive_upload/AnchorFlow_v10_1_LiteMetric'
OUT=ROOT/'results/anchorflow_v10_completed_audit'
sys.path.insert(0,str(FOLDER))
import run
from model import AnchorFlowEdge,Deploy
from data import KITTIDataset

def main():
    torch.set_num_threads(2);torch.manual_seed(123)
    model=AnchorFlowEdge().eval()
    parent=torch.load(OUT/'best.pth',map_location='cpu',weights_only=False)
    model.load_state_dict(parent['model'],strict=False)
    with torch.no_grad():
        model.phase_context[-1].weight.normal_(0,.01)
        model.dynamics.step_head.weight.normal_(0,.02)
        model.dynamics.step_head.bias.zero_()
    model.set_diagnostics(False);deploy=Deploy(model).eval()
    inputs=run.sample_inputs(torch.device('cpu'),channels_last=False)
    path=OUT/'v10_1_learned_h_open_export.onnx'
    torch.onnx.export(deploy,inputs,str(path),input_names=['rgb','sparse','mask','K'],output_names=['depth_m'],opset_version=17,dynamo=False)
    onnx.checker.check_model(onnx.load(path))
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.log_severity_level=3
    session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    ds=object.__new__(KITTIDataset);ds.root=ROOT/'data/teacher_subset_2000/kitti_bundle';ds.split,ds.teacher='val',False
    ds.rows=[r.split() for r in (ds.root/'splits/val_400.txt').read_text().splitlines()]
    cases=[]
    for seed in (0,7,42):
        torch.manual_seed(seed);x=run.sample_inputs(torch.device('cpu'),channels_last=False)
        cases.extend(((f'random_{seed}',x),(f'empty_{seed}',(x[0],x[1]*0,x[2]*0,x[3]))))
    for i in (0,199,399):
        b=ds[i];cases.append((f'real_{i}',tuple(b[k][None] for k in ('rgb','sparse','mask','K'))))
    errors={};hs={}
    with torch.inference_mode():
        for label,x in cases:
            expected=deploy(*x).numpy();actual=session.run(None,{k:v.numpy() for k,v in zip(('rgb','sparse','mask','K'),x)})[0]
            errors[label]=float(np.abs(expected-actual).max())
            assert np.isfinite(actual).all() and np.allclose(actual,expected,atol=.01,rtol=1e-4),(label,errors[label])
            model.set_diagnostics(True);o=model(*x);model.set_diagnostics(False)
            hs[label]=[float(o[f'dynamics_step_mean_{k}']) for k in (1,2)]
            print('Learned-h ONNX:',label,errors[label],hs[label],flush=True)
    assert max(v[0] for v in hs.values())-min(v[0] for v in hs.values())>1e-6,'Timestep not actually sample dependent'
    proof=json.loads((FOLDER/'local_verification.json').read_text())
    proof['onnx_learned_step_open']={'passed':True,'cases':errors,'step_means':hs,
        'nonzero_step_weights':True,'sample_dependent_h_verified':True,'device':'CPU','gpu_latency_measured':False}
    run.write_json(FOLDER/'local_verification.json',proof)

if __name__=='__main__':main()
